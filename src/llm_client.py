# src/llm_client.py
"""
Keeper LLM client with tool-intercept workflow (project-style):
- Investigator speaks -> LLM (Keeper) plans and may emit <TOOL_CALL>{...}</TOOL_CALL>
- Engine executes tool(s) (roll / sanity / clue / inventory / state persistence)
- LLM continues narration based on TOOL_RESULT (no "made-up" dice / SAN / clue)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from src.tools import normalize_proposed_check, normalize_effect_list, lookup_ci

loaded = load_dotenv()  # loads GEMINI_API_KEY


# ----------------------------
# Safe imports
# ----------------------------
try:
    from . import mechanics  # roll + sanity helpers
except Exception:
    import mechanics  # type: ignore

try:
    from . import rag_engine
except Exception:
    import rag_engine  # type: ignore

try:
    from . import state_manager
except Exception:
    import state_manager  # type: ignore
# ----------------------------
# OpenAI-compatible LLM client
# ----------------------------


def _get_openai_client():
    provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    print(f"Using LLM provider:{provider}")

    if provider == "gemini":
        base_url = os.getenv(
            "OPENAI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        api_key = os.getenv("GEMINI_API_KEY", "")
        model_default = "gemini-2.5-flash"
    elif provider == "qwen":
        if not loaded:
            print("❌ Error: .env file not found!")
        elif not os.getenv("DASHSCOPE_API_KEY"):
            print("❌ Error: DASHSCOPE_API_KEY not found in .env!")
        else:
            print("✅ Environment loaded successfully. API Key detected.")
        # Singapore (intl) default; switch to Beijing if needed
        base_url = os.getenv(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        # "qwen-turbo(-latest)" is guaranteed by Alibaba's OpenAI-compatible list
        # model_default = "qwen-turbo-latest"
        model_default = "qwen-flash"
        print(f"Using Qwen model: {model_default}")

    else:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY", "")
        model_default = "gpt-4o-mini"

    if not api_key:
        raise RuntimeError(f"Missing API key for provider={provider}")

    # Let KeeperConfig override model if set
    # os.environ.setdefault("OPENAI_MODEL", model_default)

    return OpenAI(api_key=api_key, base_url=base_url)


@dataclass
class KeeperConfig:
    model: str = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
    temperature: float = 0.8
    max_tokens: int = 900

    # RAG settings
    enable_rag: bool = True
    rag_k: int = 5
    rag_search_type: str = "mmr"
    rag_category_rules: str = "rule_system"
    rag_category_scenario: Optional[str] = None  # keeper can see spoilers if None
    rag_use_rerank: bool = True

    # Turn safety
    max_tool_calls_per_turn: int = 1


# ----------------------------
# Tool-call protocol
# ----------------------------
_TOOL_BLOCK_RE = re.compile(
    r"<TOOL_CALL>(.*?)</TOOL_CALL>", flags=re.DOTALL | re.IGNORECASE
)

_TOOL_OPEN_BLOCK_RE = re.compile(r"<TOOL_CALL>(.*)$", flags=re.DOTALL | re.IGNORECASE)

_ALT_TOOL_CALL_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\((.*?)\)\s*\]", flags=re.DOTALL)


def _parse_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    """
    Robustly parse the first JSON object in a string.
    This avoids regex '{...}' truncation when nested braces exist.
    """
    s = s.strip()

    # Common: model wraps with ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # Find the first '{'
    i = s.find("{")
    if i < 0:
        return None

    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s[i:])
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Supports:
      1) <TOOL_CALL>{ "name": "...", "arguments": {...} }</TOOL_CALL>
         - robust even with nested JSON (arguments/stakes/proposed_check)
      2) [TOOL: tool_name(arg)]  # minimal legacy format

    Returns: {"name": str, "arguments": dict}
    """
    # 1) Closed tag block
    m = _TOOL_BLOCK_RE.search(text)
    raw_block = None
    if m:
        raw_block = m.group(1)
    else:
        # 2) Unclosed tag at end (rare): <TOOL_CALL>...EOF
        m_open = _TOOL_OPEN_BLOCK_RE.search(text.strip())
        if m_open:
            raw_block = m_open.group(1)

    if raw_block:
        obj = _parse_first_json_object(raw_block)
        if isinstance(obj, dict) and "name" in obj:
            obj.setdefault("arguments", {})
            if not isinstance(obj["arguments"], dict):
                # some models may output arguments as string -> keep as {"value": ...}
                obj["arguments"] = {"value": obj["arguments"]}
            return obj

    # 3) Legacy alt format
    m2 = _ALT_TOOL_CALL_RE.search(text)
    if m2:
        name = m2.group(1).strip()
        arg = m2.group(2).strip()
        arguments: Dict[str, Any] = {}
        if arg:
            try:
                arguments = json.loads(arg)
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
            except Exception:
                arguments = {"value": arg}
        return {"name": name, "arguments": arguments}

    return None


def strip_tool_call_blocks(text: str) -> str:
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _ALT_TOOL_CALL_RE.sub("", text)
    # remove stray opening tag if any
    text = re.sub(r"<TOOL_CALL>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


# -----------------------------------------
# Tool-call gating: narrative vs rules turn
# -----------------------------------------

_TOOL_KEYWORDS_RE = re.compile(
    r"\b("
    r"roll|dice|d100|check|test|skill|stat|spot hidden|dodge|listen|library use|"
    r"sanity|san|lose\s+san|hp|damage|hurt|wound|injury|"
    r"move|enter|go\s+to|walk\s+to|run\s+to|"
    r"clue|find|discover|take|pick up|inventory|item|"
    r"flag|condition"
    r")\b",
    flags=re.IGNORECASE,
)


def need_tool_call(investigator_utterance: str, draft: str, state: dict) -> bool:
    """
    Returns True if this turn likely implies a rule-governed state change
    and should therefore require a TOOL_CALL.
    """
    # Explicit requirement in text
    if "[REQUIRES_TOOL]" in draft:
        return True

    # --- Alert-driven escalation ---
    game = state.get("game", {})
    alert_level = int(game.get("alert_level", 0))
    # Once alert is high, force rules engagement
    if alert_level >= 3:
        return True

    text = f"{investigator_utterance}\n{draft}"

    # --- Semantic MUST_CHECK triggers (playtest-friendly) ---
    t = investigator_utterance.lower()
    # A) Physical danger / combat / chase -> must resolve by rules
    if any(
        w in t
        for w in [
            "attack",
            "fight",
            "shoot",
            "stab",
            "punch",
            "kick",
            "grab",
            "dodge",
            "evade",
            "run",
            "chase",
            "jump",
            "climb",
        ]
    ):
        return True
    # B) Information gathering with meaningful consequences -> must roll
    if any(
        w in t
        for w in [
            "search",
            "examine",
            "inspect",
            "look closely",
            "check the drawer",
            "open the drawer",
            "listen",
            "eavesdrop",
            "read",
            "research",
        ]
    ):
        return True
    # C) Supernatural exposure -> must trigger SAN mechanics (roll or direct SAN effect)
    if any(
        w in t
        for w in [
            "impossible",
            "supernatural",
            "unnatural",
            "ghost",
            "moved by itself",
            "air warps",
            "whispering from nowhere",
        ]
    ):
        return True
    # If the user prompt explicitly demands a tool call, honor it.
    if re.search(
        r"\b(must|you must)\b.*\b(tool_call|tool call|resolve_action)\b",
        text,
        flags=re.I,
    ):
        return True
    # If keywords suggest checks/damage/sanity/move/items/clues etc.
    return bool(_TOOL_KEYWORDS_RE.search(text))


# ----------------------------
# Mechanics helpers
# ----------------------------
_SUCCESS_RANK = {
    "Fumble": 0,
    "Failure": 1,
    "Regular Success": 2,
    "Hard Success": 3,
    "Extreme Success": 4,
    "Critical Success": 5,
}


def _success_ge(level: str, minimum: str) -> bool:
    return _SUCCESS_RANK.get(level, 0) >= _SUCCESS_RANK.get(minimum, 0)


def _parse_sanity_pair(loss_str: str) -> tuple[str, str]:
    """
    Input:
      - "0/1d6"  -> ("0", "1d6")
      - "1/1d6"  -> ("1", "1d6")
      - "1d4"    -> ("1d4", "1d4")   # no check provided -> same for success/fail
      - "5"      -> ("5", "5")
    """
    s = loss_str.strip().lower()

    if "/" in s:
        a, b = s.split("/", 1)
        a, b = a.strip(), b.strip()
        # allow empty like "/1d6" or "1/" (optional)
        a = a if a != "" else "0"
        b = b if b != "" else a
        return a, b

    return s, s


# ----------------------------
# Engine tool: resolve_action
# ----------------------------
def resolve_action(
    *,
    scene_id: str,
    intent: str,
    proposed_check: Optional[Dict[str, Any]] = None,
    stakes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified "referee" tool:
      - optionally roll d100 for a skill/stat check
      - apply stakes: clue/item/sanity/hp/alert/move/flags
      - persist state
      - return a result bundle for narration

    Input example:
    {
      "scene_id": "storage_room",
      "intent": "search_drawer",
      "proposed_check": {"type":"skill","name":"Spot Hidden","difficulty":"regular"},
      "stakes": {
          "on_success": ["clue:diary_page_1", "item:Brass Key"],
          "on_failure": ["alert:+1", "sanity:0/1d6"]
      }
    }
    """
    st = state_manager.load_state()
    new_turn = state_manager.increment_turn()
    # keep in-memory state consistent (important because we save `st` later)
    st.setdefault("game", {})["turn_count"] = new_turn

    inv = st.get("investigator", {})
    skills = inv.get("skills", {})
    stats = inv.get("stats", {})

    roll_info = None
    check_result = "no_check"
    success_level = None
    # --- Backward-compatible: proposed_check may be a plain string like "Lockpick" ---
    if isinstance(proposed_check, str) and proposed_check.strip():
        proposed_check = {
            "type": "skill",
            "name": proposed_check.strip(),
            "difficulty": "regular",
        }
    # 1) Decide and execute check
    # --- Fallback: infer proposed_check from intent when missing (playtest mode) ---
    if not proposed_check:
        it = (intent or "").lower()
        if any(w in it for w in ["dodge", "evade", "avoid"]):
            proposed_check = {"type": "skill", "name": "Dodge", "difficulty": "regular"}
        elif any(w in it for w in ["attack", "fight", "strike", "punch", "kick"]):
            proposed_check = {
                "type": "skill",
                "name": "Fight Brawl",
                "difficulty": "regular",
            }
        elif any(w in it for w in ["search", "examine", "inspect", "look", "spot"]):
            proposed_check = {
                "type": "skill",
                "name": "Spot Hidden",
                "difficulty": "regular",
            }
        elif any(w in it for w in ["listen", "eavesdrop"]):
            proposed_check = {
                "type": "skill",
                "name": "Listen",
                "difficulty": "regular",
            }
        elif any(w in it for w in ["read", "research", "library"]):
            proposed_check = {
                "type": "skill",
                "name": "Library Use",
                "difficulty": "regular",
            }
        # supernatural: often best handled via stakes.effects sanity token (no check)
    proposed_check = normalize_proposed_check(proposed_check)

    if proposed_check:
        ctype = (proposed_check.get("type") or "").lower()
        cname = proposed_check.get("name")
        difficulty = proposed_check.get("difficulty", "regular")

        # --- normalize common LLM formatting errors ---
        # 1) "Spot Hidden / regular" -> "Spot Hidden"
        if isinstance(cname, str) and "/" in cname:
            cname = cname.split("/", 1)[0].strip()
        # 2) If model accidentally puts difficulty into name or mixes stat/skill
        STAT_NAMES = {"STR", "CON", "SIZ", "DEX", "APP", "INT", "POW", "EDU"}
        if isinstance(cname, str) and cname.upper() in STAT_NAMES:
            # If it looks like a stat, treat as stat even if model said "skill"
            ctype = "stat"
            cname = cname.upper()
        # 3) Skill name normalization: map common incorrect names to correct ones
        if isinstance(cname, str) and ctype == "skill":
            skill_name_map = {
                "perception": "Spot Hidden",  # Common error: use Spot Hidden for visual perception
                "observation": "Spot Hidden",
                "awareness": "Spot Hidden",
                "investigation": "Spot Hidden",  # Investigation is not a skill, use Spot Hidden or Library Use
            }
            cname_lower = cname.strip().lower()
            if cname_lower in skill_name_map:
                mapped_name = skill_name_map[cname_lower]
                # Only map if the mapped skill exists
                if lookup_ci(skills, mapped_name) is not None:
                    print(
                        f"[DEBUG] SKILL_NORMALIZATION: mapped '{cname}' -> '{mapped_name}'"
                    )
                    cname = mapped_name
                else:
                    # If mapped skill doesn't exist, try to find a similar one
                    # For "Perception" -> try "Listen" if context suggests hearing
                    if cname_lower == "perception":
                        # Try Listen as fallback (common for listening contexts)
                        if lookup_ci(skills, "Listen") is not None:
                            print(
                                f"[DEBUG] SKILL_NORMALIZATION: mapped 'Perception' -> 'Listen' (fallback)"
                            )
                            cname = "Listen"
        # 4) Clean difficulty if the model emits weird values
        difficulty = (difficulty or "regular").lower().strip()
        if difficulty not in ("regular", "hard", "extreme"):
            difficulty = "regular"

        target = None
        if ctype == "skill" and cname:
            target = lookup_ci(skills, str(cname))
            print(
                f"[DEBUG] SKILL_LOOKUP: requested={cname!r}, target={target}, available_keys_sample={list(skills.keys())[:10]}"
            )
        elif ctype == "stat" and cname:
            target = lookup_ci(stats, str(cname))
            print(
                f"[DEBUG] SKILL_LOOKUP: requested={cname!r}, target={target}, available_keys_sample={list(skills.keys())[:10]}"
            )

        if target is None:
            # unknown check -> treat as failure with explanation
            roll_info = {"dice": "1d100", "value": None, "target": None}
            check_result = f"invalid_check(no {ctype}:{cname})"
            success_level = "Failure"
            print(f"[DEBUG] CHECK: no_check (no valid {ctype}:{cname})")
        else:
            roll, level = mechanics.check_skill_success(target, difficulty=difficulty)
            roll_info = {
                "dice": "1d100",
                "value": int(roll),
                "target": int(target),
                "difficulty": difficulty,
                "name": cname,
                "type": ctype,
            }
            success_level = level
            check_result = (
                "success" if "Success" in level and level != "Failure" else "fail"
            )
            # fumble counts as fail
            if level == "Fumble":
                check_result = "fail"
            print(
                "[DEBUG] CHECK:",
                {
                    "type": ctype,
                    "name": cname,
                    "difficulty": difficulty,
                    "target": target,
                    "roll": roll,
                    "roll_level": level,
                },
            )
    else:
        print("[DEBUG] CHECK: no_check (no proposed_check)")

    # Helper: choose list of effects
    stakes = stakes or {}
    effects: List[Dict[str, Any]] = []
    state_patch: List[Dict[str, Any]] = []

    def apply_effect_token(token: str) -> None:
        token = token.strip()
        if not token:
            return

        # clue:xxx
        if token.startswith("clue:"):
            clue_id = token.split(":", 1)[1].strip()
            clues = st.setdefault("game", {}).setdefault("clues_found", [])
            applied = False
            if clue_id and clue_id not in clues:
                clues.append(clue_id)
                applied = True
            effects.append({"op": "add_clue", "clue_id": clue_id, "applied": applied})
            if applied:
                state_patch.append({"path": "game.clues_found", "add": clue_id})
            print(f"[DEBUG] Applied clue: {clue_id}, applied={applied}")
            return

        # item:xxx
        if token.startswith("item:"):
            item = token.split(":", 1)[1].strip()
            inv = st.setdefault("investigator", {})
            inv_list = inv.setdefault("inventory", [])
            applied = False
            if item and item not in inv_list:
                inv_list.append(item)
                applied = True
            effects.append({"op": "add_item", "item": item, "applied": applied})
            if applied:
                state_patch.append({"path": "investigator.inventory", "add": item})
            print(f"[DEBUG] Applied item: {item}, applied={applied}")
            return

        # sanity:0/1d6 or sanity:1d4 or sanity:5
        if token.startswith("sanity:"):
            loss_str = token.split(":", 1)[1].strip()
            cur = int(st["investigator"]["sanity"]["current"])

            a, b = _parse_sanity_pair(loss_str)

            # If check happened and succeeded => use a, else use b.
            # If no check => treat as fail-loss (b).
            if success_level and _success_ge(success_level, "Regular Success"):
                chosen = a
            else:
                chosen = b

            new_san, loss_val, temp_ins = mechanics.calculate_sanity_loss(cur, chosen)
            delta = int(new_san) - cur  # negative or 0

            st["investigator"]["sanity"]["current"] = int(new_san)
            if temp_ins:
                st["investigator"]["sanity"]["temp_insanity"] = True

            effects.append(
                {
                    "op": "sanity_change",
                    "delta": delta,
                    "loss": int(loss_val),
                    "chosen": chosen,
                    "temp_insanity": bool(temp_ins),
                    "rule": loss_str,
                    "success_level": success_level,
                }
            )
            state_patch.append({"path": "investigator.sanity.current", "delta": delta})
            if temp_ins:
                state_patch.append(
                    {"path": "investigator.sanity.temp_insanity", "set": True}
                )
            print(
                f"[DEBUG] Applied sanity loss_str: {loss_str}, loss: {chosen}, loss_val={loss_val}, new_san={new_san}, temp_insanity={temp_ins}"
            )
            return

        # hp:-1 / hp:+2
        if token.startswith("hp:"):
            delta_str = token.split(":", 1)[1].strip()
            try:
                delta = int(delta_str)
            except Exception:
                delta = int(delta_str.replace("+", ""))

            hp = st.setdefault("investigator", {}).setdefault(
                "hp", {"current": 0, "max": 0}
            )
            cur_hp = int(hp.get("current", 0))
            max_hp = int(hp.get("max", cur_hp))
            new_hp = max(0, min(max_hp, cur_hp + delta))
            hp["current"] = new_hp

            effects.append({"op": "hp_change", "delta": new_hp - cur_hp})
            state_patch.append(
                {"path": "investigator.hp.current", "delta": new_hp - cur_hp}
            )
            print(f"[DEBUG] Applied HP change: {delta}, new_hp={new_hp}")
            return

        # alert:+1
        if token.startswith("alert:"):
            delta_str = token.split(":", 1)[1].strip()
            try:
                delta = int(delta_str.replace("+", ""))
            except Exception:
                return

            game = st.setdefault("game", {})
            game["alert_level"] = int(game.get("alert_level", 0)) + delta

            effects.append({"op": "alert_change", "delta": delta})
            state_patch.append({"path": "game.alert_level", "set": game["alert_level"]})
            print(
                f"[DEBUG] Applied alert change: {delta}, new_alert_level={game['alert_level']}"
            )
            return

        # move:LocationName
        if token.startswith("move:"):
            loc = token.split(":", 1)[1].strip()
            st.setdefault("game", {})["location"] = loc
            effects.append({"op": "move", "to": loc})
            state_patch.append({"path": "game.location", "set": loc})
            print(f"[DEBUG] Applied move to: {loc}")
            return

        # flag:temp_insanity=true
        if token.startswith("flag:"):
            expr = token.split(":", 1)[1].strip()
            if "=" in expr:
                k, v = expr.split("=", 1)
                k = k.strip()
                v = v.strip().lower()
                val: Any = (
                    True
                    if v in ("true", "1", "yes")
                    else False
                    if v in ("false", "0", "no")
                    else v
                )
                st.setdefault("game", {})[k] = val
                effects.append({"op": "set_flag", "key": k, "value": val})
                state_patch.append({"path": f"game.{k}", "set": val})
            print(
                f"[DEBUG] Applied flag: {expr}, effects: {effects[-1] if effects else None}, state_patch: {state_patch[-1] if state_patch else None}"
            )
            return

        # Unknown token
        effects.append({"op": "unknown_effect", "raw": token})
        print(f"[DEBUG] Unknown effect token: {token}")

    # 2) Apply stakes based on outcome
    # Determine which list to apply
    outcome = "no_check"
    if success_level:
        outcome = (
            "success" if _success_ge(success_level, "Regular Success") else "failure"
        )
        if success_level == "Fumble":
            outcome = "failure"

    chosen_effects: List[str] = []

    # --- Enforce string-token-only stakes (drop non-strings) ---
    def _toklist(x) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x.strip()] if x.strip() else []
        if not isinstance(x, list):
            return []
        out: List[str] = []
        for it in x:
            if isinstance(it, str) and it.strip():
                out.append(it.strip())
        return out

    if outcome == "success":
        chosen_effects = _toklist(stakes.get("on_success", []))
    elif outcome == "failure":
        chosen_effects = _toklist(stakes.get("on_failure", []))
    else:
        chosen_effects = _toklist(stakes.get("effects", []))

    print(f"[DEBUG] success_level: {success_level}  outcome: {outcome}")
    print("[DEBUG] chosen_effects:", chosen_effects)

    # --- Default stakes injection (guarantee meaningful outcomes) ---
    stakes = stakes or {"on_success": [], "on_failure": [], "effects": []}
    stakes.setdefault("on_success", [])
    stakes.setdefault("on_failure", [])
    stakes.setdefault("effects", [])

    # If a check happened but stakes are empty, inject conservative defaults
    if success_level and (not stakes["on_success"]) and (not stakes["on_failure"]):
        it = (intent or "").lower()

        # Info gathering checks: success gives clue, failure raises alert
        if any(
            w in it
            for w in ["search", "examine", "inspect", "listen", "read", "research"]
        ):
            stakes["on_success"] = ["clue:minor_detail"]
            stakes["on_failure"] = ["alert:+1"]

        # Physical danger checks: failure costs HP
        elif any(
            w in it
            for w in ["dodge", "attack", "fight", "run", "chase", "jump", "climb"]
        ):
            stakes["on_success"] = ["alert:-1"]
            stakes["on_failure"] = ["hp:-2", "alert:+1"]

    # Supernatural exposure: if no explicit stakes, apply SAN via effects
    if not success_level and (not stakes["effects"]):
        it = (intent or "").lower()
        if any(w in it for w in ["supernatural", "impossible", "unnatural", "ghost"]):
            stakes["effects"] = ["sanity:0/1d4", "alert:+1"]

    for tok in chosen_effects:
        if isinstance(tok, str):
            apply_effect_token(tok)
    # 3) Save final state
    state_manager.save_state(st)
    # 4) Narration hints (for LLM style)
    temp_insanity = bool(st["investigator"]["sanity"].get("temp_insanity", False))
    narration_hints = {
        "scene_id": scene_id,
        "intent": intent,
        "outcome": outcome,
        "temp_insanity": temp_insanity,
        "tone": "tense" if outcome != "success" else "suspenseful",
        "reveal": "partial" if outcome == "success" else "none",
        "next_hooks": [
            "sound_in_distance",
            "cold_draft",
            "unexpected_detail",
            "shadowy_figure",
        ],
    }

    return {
        "roll": roll_info,
        "success_level": success_level,
        "check_result": check_result,
        "effects": effects,
        "state_patch": state_patch,
        "state_snapshot": {
            "location": st.get("game", {}).get("location"),
            "hp": st.get("investigator", {}).get("hp"),
            "sanity": st.get("investigator", {}).get("sanity"),
            "clues_found": st.get("game", {}).get("clues_found", []),
            "inventory": st.get("investigator", {}).get("inventory", []),
            "turn_count": st.get("game", {}).get("turn_count", 0),
        },
        "narration_hints": narration_hints,
    }


# ----------------------------
# Keeper LLM Client
# ----------------------------
class KeeperLLMClient:
    def __init__(self, cfg: Optional[KeeperConfig] = None):
        self.cfg = cfg or KeeperConfig()
        self._client = _get_openai_client()

        # Lazy RAG engine
        self._rag = None

    # ---- RAG ----
    def _get_rag_engine(self):
        if self._rag is None:
            # rag_engine.py provides module singleton get_engine()
            self._rag = rag_engine.get_engine()
        return self._rag

    def rag_retrieve(
        self, query: str, *, allow_spoilers: bool = True
    ) -> List[Dict[str, Any]]:
        if not self.cfg.enable_rag:
            return []
        eng = self._get_rag_engine()

        # Keeper can see scenario (spoilers) if allow_spoilers=True and category=None
        category = None if allow_spoilers else self.cfg.rag_category_rules

        try:
            # Use engine.retrieve() so it can apply rerank and formatting
            return eng.retrieve(
                query,
                search_type=self.cfg.rag_search_type,
                k=self.cfg.rag_k,
                category=category,
                use_rerank=self.cfg.rag_use_rerank,
                return_raw_docs=False,
            )
        except Exception:
            # If DB isn't built yet, can call build_vector_database elsewhere.
            return []

    # ---- Prompt building ----
    def build_fallback_narration(self, tool_result: dict) -> str:
        effects = tool_result.get("effects", [])
        snap = tool_result.get("state_snapshot", {})

        lines = []
        perceptions = []

        # 1) Translate effects into neutral narrative fragments
        for e in effects:
            op = e.get("op")

            if op == "hp_change":
                delta = e.get("delta", 0)
                if delta < 0:
                    lines.append(f"(HP {delta}. Pain flares through your body.)")
                    perceptions.append("a sharp jolt of pain")
                elif delta > 0:
                    lines.append(f"(HP +{delta}. You steady yourself.)")
                    perceptions.append("your breathing slowly steadies")

            elif op == "sanity_change":
                delta = e.get("delta", 0)
                if delta < 0:
                    lines.append(f"(SAN {delta}. Your nerves fray.)")
                    perceptions.append("a creeping sense of unease")

            elif op == "move":
                to = e.get("to")
                lines.append(f"(You move to {to}.)")
                perceptions.append(f"the atmosphere of {to}")

            elif op == "add_clue":
                perceptions.append("a troubling detail that catches your eye")

            elif op == "set_flag":
                perceptions.append("the feeling that something has shifted")

        # 2) Compose minimal Keeper-style narration
        text = "You barely have time to react. The consequences are immediate.\n\n"

        if perceptions:
            text += "You notice " + ", ".join(perceptions[:2]) + ".\n\n"

        if lines:
            text += "\n".join(lines) + "\n\n"

        text += "What do you do next?"
        return text

    def _build_system_prompt(self, state: Dict[str, Any]) -> str:
        inv = state.get("investigator", {})
        game = state.get("game", {})

        temp_ins = bool(inv.get("sanity", {}).get("temp_insanity", False))
        if (
            inv.get("sanity", {}).get("current") is not None
            and inv.get("sanity", {}).get("max") is not None
        ):
            san_ratio = inv.get("sanity", {}).get("current") / inv.get(
                "sanity", {}
            ).get("max")

        # Get available skills and stats for prompt
        skills = inv.get("skills", {})
        stats = inv.get("stats", {})
        skill_names = list(skills.keys()) if skills else []
        stat_names = list(stats.keys()) if stats else []

        base = f"""
You are the Keeper (Game Master) for a Call of Cthulhu-style tabletop RPG set in 1920s Boston.

Your goals:
- Be atmospheric and responsive.
- Maintain rules correctness: NEVER invent dice results, SAN/HP changes, clues, or inventory changes.
- Mechanics must be executed only via tools. If you need a roll or a state change, emit exactly one TOOL_CALL.
- If any state might change, you MUST emit exactly one TOOL_CALL. Otherwise narrate without any tool.
- After a tool result is returned, you MUST narrate based on that result, summarize state changes, and offer next actions.

Current game state (summary):
- Location: {game.get("location")}
- HP: {inv.get("hp")}
- SAN: {inv.get("sanity")}
- Inventory: {inv.get("inventory")}
- Clues found: {game.get("clues_found")}
- Turn: {game.get("turn_count")}

AVAILABLE SKILLS (use exact names in proposed_check):
{", ".join(skill_names) if skill_names else "None"}

AVAILABLE STATS (use exact names in proposed_check):
{", ".join(stat_names) if stat_names else "None"}

IMPORTANT: When the investigator uses "(Listen)" or mentions listening/hearing, use the skill "Listen" (NOT "Perception").
When they mention seeing/observing/spotting, use "Spot Hidden" (NOT "Perception" or "Observation").
Perception is NOT a valid skill name in Call of Cthulhu.

TOOL_CALL RULE (STRICT):
- If ANY mechanical change is needed (roll/check, HP/SAN change, inventory/clue/flag change, or moving location),
  output exactly one TOOL_CALL block and nothing else.

TOOL_CALL FORMAT:
<TOOL_CALL>{{"name":"resolve_action","arguments":{{...}}}}</TOOL_CALL>

ARGUMENTS REQUIRED:
- intent (short)
- scene_id (location only)
- proposed_check (object or null)
- stakes: {{"on_success":[], "on_failure":[], "effects":[]}}

ABSOLUTE:
- No extra text outside TOOL_CALL.
- Valid JSON only. Double quotes. Closed </TOOL_CALL>.
- Do NOT put dict/object inside stakes.

ALLOWED EFFECT TOKENS (strings only):
- move:<Location>
- hp:+N or hp:-N
- sanity:0/1d4  (or sanity:-1)
- alert:+1 / alert:-1
- clue:<id>
- item:<name>
- flag:<key>=<true|false|string>


Narration RULES:
- After receiving TOOL_RESULT, narrate based ONLY on that result.
- DO NOT embed tokens inside narration text.
- When temp_insanity=true, you may add subtle distortions / paranoia in narration;
  BUT still must not invent mechanics: any SAN changes must come from TOOL_RESULT.
- When the atmosphere becomes tense (high alert), you should stop hand-waving narration and rely on rule resolution.
- If you cannot generate a full narration, at minimum summarize:
  what changed (HP/SAN/location/etc.) and what the investigator perceives.

Navigation & pacing:
- If the situation is not locked in combat or immediate danger, include at least ONE exploration/movement option
  (e.g., "Move to the Hallway and listen at the next door", "Check the Kitchen", "Go upstairs").
- If danger is immediate (combat/hostile NPC/high alert), exploration options must be tactical
  (e.g., "Retreat to the hallway", "Take cover behind the dresser", "Run for the front door").
- Do NOT invent new rooms; only suggest locations that exist in the scenario or have been revealed by context/RAG.

""".strip()
        if temp_ins:
            base += "\n\nTemp insanity is active. Increase dread and perceptual uncertainty, but stay consistent with TOOL_RESULT."
            print(f"[DEBUG] Temp insanity is active for investigator")
        if san_ratio is not None and san_ratio < 0.25:
            base += f"\nSanity ratio: {san_ratio:.2f}"
            print(f"[DEBUG] sanity ratio={san_ratio:.2f}")
        return base

    def _format_rag_context(self, rag_results: List[Dict[str, Any]]) -> str:
        if not rag_results:
            print(f"[DEBUG] No reletive docs from DB")
            return ""
        print(f"[DEBUG] RAG get {len(rag_results)} docs from DB")
        lines = [
            "[RAG_CONTEXT] Relevant excerpts (use as grounding; cite by source/page if needed):"
        ]
        for i, r in enumerate(rag_results, 1):
            src = r.get("source")
            page = r.get("page")
            cat = r.get("category")
            txt = (r.get("text") or "").strip().replace("\n", " ")
            if len(txt) > 360:
                txt = txt[:360] + "..."
            lines.append(f"{i}. ({cat}) {src} p{page}: {txt}")
        return "\n".join(lines)

    # ---- LLM call ----
    def _chat(self, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ---- Public API: one full turn ----
    def run_turn(self, investigator_utterance: str, force_tool: bool = False) -> str:
        """
        One full turn:
          - Keeper may propose and emit TOOL_CALL
          - Execute tool
          - Feed TOOL_RESULT back
          - Keeper narrates outcome + next step
        Returns final text for the Investigator.
        """
        state = state_manager.load_state()

        # RAG query: combine intent + location for better hits
        location = state.get("game", {}).get("location", "")
        rag_q = (
            f"{investigator_utterance}\n"
            f"Location: {location}\n"
            f"Retrieve: room details, relevant rules (like skill/stats/sanity, only if needed), and any clues/hazards."
        )

        rag_ctx = self.rag_retrieve(rag_q, allow_spoilers=True)
        rag_text = self._format_rag_context(rag_ctx)

        system = self._build_system_prompt(state)

        # First assistant response (plan + maybe tool-call)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{rag_text}\n\n[INVESTIGATOR]\n{investigator_utterance}".strip(),
            },
        ]
        draft = self._chat(messages)
        print(f"[DEBUG] chat respond-TOOL_CALL: {draft}")
        tool_call = extract_tool_call(draft)

        print("[DEBUG] tool_call parsed:", bool(tool_call))
        if tool_call:
            print("[DEBUG] tool_call name:", tool_call.get("name"))
            print(
                "[DEBUG] tool_call args keys:",
                list((tool_call.get("arguments") or {}).keys()),
            )

        # If no tool call:
        # - narrative turns are allowed (return draft)
        # - but if this turn implies rules/state change, retry once with strict instruction
        if not tool_call:
            current_state = state_manager.load_state()
            must_tool = force_tool or need_tool_call(
                investigator_utterance, draft, current_state
            )
            print("[DEBUG] No tool call emitted by Keeper; need_tool_call =", must_tool)

            if not must_tool:
                return draft.strip()

            # Retry once: require ONLY a tool call block
            messages_retry = messages + [
                {
                    "role": "user",
                    "content": (
                        "You MUST output exactly ONE tool call now.\n"
                        "Output ONLY a single block in this exact format:\n"
                        '<TOOL_CALL>{"name":"resolve_action","arguments":{...}}</TOOL_CALL>\n'
                        "No narration. No extra text."
                    ),
                }
            ]
            draft_retry = self._chat(messages_retry)
            tool_call = extract_tool_call(draft_retry)

            print("[DEBUG] tool_call parsed after retry:", bool(tool_call))
            if tool_call:
                print("[DEBUG] tool_call name:", tool_call.get("name"))
                print(
                    "[DEBUG] tool_call args keys:",
                    list((tool_call.get("arguments") or {}).keys()),
                )
                draft = draft_retry  # Use retry output as the tool-call carrier
            else:
                # If still no tool call, fall back to the original narrative draft
                return draft.strip()

        # Enforce tool-call cap
        tool_calls_used = 0
        tool_result = None

        # Execute exactly one tool call
        if tool_call.get("name") == "resolve_action":
            tool_calls_used += 1
            args = tool_call.get("arguments", {}) or {}

            # --- normalize common LLM argument aliases ---
            if "intent" not in args and "action" in args:
                args["intent"] = args["action"]
            if "scene_id" not in args:
                if "location" in args:
                    args["scene_id"] = args["location"]
                elif "destination" in args:
                    args["scene_id"] = args["destination"]
                elif "target" in args:
                    args["scene_id"] = args["target"]

            # Minimal defaults to avoid LLM forgetting required fields
            args.setdefault("scene_id", location or "unknown_scene")
            args.setdefault("intent", "freeform")
            tool_result = resolve_action(
                scene_id=str(args.get("scene_id")),
                intent=str(args.get("intent")),
                proposed_check=normalize_proposed_check(args.get("proposed_check")),
                stakes=args.get("stakes"),
            )
        else:
            tool_result = {"error": f"Unknown tool: {tool_call.get('name')}"}

        # Feed TOOL_RESULT back and force narration continuation
        cleaned = strip_tool_call_blocks(draft)

        messages2 = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{rag_text}\n\n[INVESTIGATOR]\n{investigator_utterance}".strip(),
            },
            {"role": "assistant", "content": cleaned},
            {
                "role": "user",
                "content": f"[TOOL_RESULT]\n{json.dumps(tool_result, ensure_ascii=False)}\n\nNow continue: narrate based ONLY on TOOL_RESULT, summarize state changes, and offer 2-4 next actions.",
            },
        ]

        final = self._chat(messages2)
        final = strip_tool_call_blocks(final).strip()
        # print(f"[DEBUG] chat respond-After TOOL_CALL: {draft}")

        if not final:
            final = self.build_fallback_narration(tool_result)

        return final
