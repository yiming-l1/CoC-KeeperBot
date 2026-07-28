import re
from src.state_manager import load_state
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"\b(?:clue|item|sanity|hp|alert|move|flag):[^\s,\]]+")


def normalize_proposed_check(proposed_check):
    """
    Accepts dict or sloppy LLM outputs (str/list) and normalizes to:
      {"type":"skill"|"stat", "name":..., "difficulty":"regular|hard|extreme"}
    Returns None if cannot parse.
    """
    if proposed_check is None:
        return None

    # Already correct
    if isinstance(proposed_check, dict):
        if "type" in proposed_check and "name" in proposed_check:
            proposed_check.setdefault("difficulty", "regular")
            return proposed_check
        # Allow dict-like but missing fields
        t = (proposed_check.get("type") or proposed_check.get("kind") or "").lower()
        n = (
            proposed_check.get("name")
            or proposed_check.get("skill")
            or proposed_check.get("stat")
        )

        raw_d = proposed_check.get("difficulty", "regular")
        # Accept int or str; normalize to one of: regular/hard/extreme
        if isinstance(raw_d, (int, float)):
            # 1=regular, 2=hard, 3=extreme (common compact encoding)
            raw_d = {1: "regular", 2: "hard", 3: "extreme"}.get(int(raw_d), "regular")
        elif raw_d is None:
            raw_d = "regular"
        else:
            raw_d = str(raw_d)

        d = raw_d.strip().lower()
        if d in ("1", "reg", "normal"):
            d = "regular"
        elif d in ("2", "hard"):
            d = "hard"
        elif d in ("3", "extreme"):
            d = "extreme"
        else:
            d = "regular"
        proposed_check["difficulty"] = d

        if t and n:
            return {"type": t, "name": n, "difficulty": d}
        return None

    # List/tuple: ["skill", "Dodge", "regular"]
    if isinstance(proposed_check, (list, tuple)) and len(proposed_check) >= 2:
        t = str(proposed_check[0]).strip().lower()
        n = str(proposed_check[1]).strip()
        d = (
            str(proposed_check[2]).strip().lower()
            if len(proposed_check) >= 3
            else "regular"
        )
        if t in ("skill", "stat") and n:
            return {"type": t, "name": n, "difficulty": d}
        return None

    # String: "skill / Dodge / regular" or "stat:POW regular" or "Dodge"
    if isinstance(proposed_check, str):
        s = proposed_check.strip()

        # Try JSON in string
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                return normalize_proposed_check(obj)
            except Exception:
                print("[DEBUG] Failed to parse proposed_check JSON:", s)
                pass

        # Common separator "/"
        parts = [p.strip() for p in s.replace("\\", "/").split("/") if p.strip()]
        if len(parts) >= 2:
            t = parts[0].lower()
            n = parts[1]
            d = parts[2].lower() if len(parts) >= 3 else "regular"
            if t in ("skill", "stat") and n:
                return {"type": t, "name": n, "difficulty": d}

        # "skill:Dodge regular"
        m = re.match(
            r"^(skill|stat)\s*[:\-]\s*([A-Za-z0-9 ()]+)\s*(regular|hard|extreme)?$",
            s,
            re.I,
        )
        if m:
            t = m.group(1).lower()
            n = m.group(2).strip()
            d = (m.group(3) or "regular").lower()
            return {"type": t, "name": n, "difficulty": d}

        # If only a skill name is provided, assume skill + regular
        if s:
            return {"type": "skill", "name": s, "difficulty": "regular"}

    return None


def normalize_effect_list(x):
    """
    Convert various sloppy formats into list[str] tokens.
    Accepts:
      - list[str]
      - dict like {"hp":-2} or {"clue":"id"} -> tokens
      - string containing tokens -> extracted tokens
    """
    if x is None:
        return []

    if isinstance(x, list):
        out = []
        for it in x:
            if isinstance(it, str):
                out.append(it.strip())
            elif isinstance(it, dict):
                out.extend(normalize_effect_list(it))
            else:
                out.extend(normalize_effect_list(str(it)))
        return [t for t in out if t]

    if isinstance(x, dict):
        out = []
        # common keys
        if "clue" in x:
            out.append(f"clue:{x['clue']}")
        if "item" in x:
            out.append(f"item:{x['item']}")
        if "sanity" in x:
            out.append(f"sanity:{x['sanity']}")
        if "hp" in x:
            v = x["hp"]
            out.append(f"hp:{v}")
        if "alert" in x:
            v = x["alert"]
            out.append(f"alert:+{v}" if isinstance(v, int) and v > 0 else f"alert:{v}")
        if "move" in x:
            out.append(f"move:{x['move']}")
        if "flag" in x:
            # flag could be dict {"corbitt_influence":true}
            if isinstance(x["flag"], dict):
                for k, v in x["flag"].items():
                    out.append(f"flag:{k}={str(v).lower()}")
            else:
                out.append(f"flag:{x['flag']}")
        # also extract tokens from any narration
        if "narration" in x and isinstance(x["narration"], str):
            out.extend(_TOKEN_RE.findall(x["narration"]))
        return [t for t in out if t]

    if isinstance(x, str):
        return [t.strip() for t in _TOKEN_RE.findall(x)]

    return []


def print_state(title: str):
    state = load_state()
    inv = state.get("investigator", {})
    game = state.get("game", {})

    print(f"\n--- STATE AFTER: {title} ---")
    print("Location:", game.get("location"))
    print("Turn:", game.get("turn_count"))
    print("HP:", inv.get("hp"))
    print("SAN:", inv.get("sanity"))
    print("Inventory:", inv.get("inventory"))
    print("Clues:", game.get("clues_found"))

    flags = {
        k: v
        for k, v in game.items()
        if k not in ("location", "clues_found", "turn_count")
    }
    if flags:
        print("Game flags:", flags)


def lookup_ci(d: Dict[str, Any], key: str) -> Optional[int]:
    """Case-insensitive dict lookup for skill/stat names."""
    if not d or not key:
        return None
    if key in d:
        try:
            return int(d[key])
        except Exception:
            return None
    lk = key.lower()
    for k, v in d.items():
        if str(k).lower() == lk:
            try:
                return int(v)
            except Exception:
                return None
    return None
