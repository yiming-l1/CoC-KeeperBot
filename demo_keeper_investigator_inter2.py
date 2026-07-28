#!/usr/bin/env python3
"""
Two-agent playtest (rules-only RAG, no spoilers):

- Keeper: KeeperLLMClient (uses TOOL_CALL + state updates)
- InvestigatorAgent: uses the SAME model backend via KeeperLLMClient._chat(),
  but NEVER uses tools. It generates a single player action each turn.

RAG:
- InvestigatorAgent uses engine.retrieve(test_retriever, category="rule_system", use_rerank=True, ...)
  and injects the returned snippets as [ARG] ... [/ARG] into its prompt.
- Keeper also receives a smaller rules-only [ARG] prefix for better rule correctness.

This improves "demo-like" playtests:
- More consistent checks/effects (rules-guided)
- No scenario spoilers (rules only)

Run:
python demo_keeper_investigator_inter2.py --start-room "Kitchen" >> ./log/demo_keeper_investigator_inter2.log
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from src.llm_client import KeeperLLMClient, KeeperConfig
from src.state_manager import load_state, save_state, DEFAULT_STATE
from src.rag_engine import build_vector_database, get_engine


# ----------------------------
# Investigator Agent
# ----------------------------


@dataclass
class InvestigatorConfig:
    temperature: float = 0.8
    max_tokens: int = 450


class InvestigatorAgent:
    """
    Uses KeeperLLMClient._chat() as a generic chat API (same model),
    but with a player-style system prompt. Never outputs TOOL_CALL.
    """

    def __init__(self, chat_client: KeeperLLMClient, engine, cfg: InvestigatorConfig):
        self.client = chat_client
        self.engine = engine
        self.cfg = cfg

    def _format_docs(self, docs: List[Any], k: int = 6) -> str:
        parts: List[str] = []
        for d in (docs or [])[:k]:
            if isinstance(d, str):
                parts.append(d.strip())
            else:
                parts.append(getattr(d, "page_content", "").strip())
        snippet = "\n".join([p for p in parts if p]).strip()
        if not snippet:
            return ""
        return f"[ARG]\n{snippet}\n[/ARG]\n"

    def build_rule_arg(
        self,
        test_retriever: str,
        *,
        search_type: str,
        return_raw_docs: bool,
    ) -> str:
        # IMPORTANT: rules-only retrieval, rerank on
        docs = self.engine.retrieve(
            test_retriever,
            category="rule_system",
            use_rerank=True,
            search_type=search_type,
            return_raw_docs=return_raw_docs,
        )
        return self._format_docs(docs, k=6)

    def propose_action(
        self,
        *,
        state: Dict[str, Any],
        keeper_last: str,
        turn_index: int,
        search_type: str,
        return_raw_docs: bool,
    ) -> str:
        game = state.get("game", {}) or {}
        inv = state.get("investigator", {}) or {}
        loc = game.get("location", "")
        alert = game.get("alert_level", 0)
        hp = inv.get("hp", {})
        san = inv.get("sanity", {})

        # This is the exact "test_retriever" arg you asked for.
        # It asks ONLY for rules (checks/difficulty/effects phrasing), no scenario content.
        test_retriever = (
            "RULES ONLY (no spoilers). "
            "Retrieve Call of Cthulhu rules and guidance relevant to the next investigator action: "
            "when a check is needed, which skill/stat fits, typical difficulty (regular/hard/extreme), "
            "and what consequences are appropriate (HP/SAN/alert/flags) without inventing scenario facts. "
            f"Context: location={loc}, alert_level={alert}, HP={hp}, SAN={san}. "
            f"Keeper said: {keeper_last[-600:]}"
        )
        rules_arg = self.build_rule_arg(
            test_retriever,
            search_type=search_type,
            return_raw_docs=return_raw_docs,
        )

        system = f"""
You are the Investigator (player) in a Call of Cthulhu-style tabletop RPG.

Constraints:
- Output ONE concrete next action in 1–3 sentences, first-person.
- Avoid spoilers: do NOT invent room contents, clues, NPC motives, or hidden facts.
- You may ONLY use the RULE snippets inside [ARG]...[/ARG] to decide what kind of action to take
  and when to hint a check (e.g., "(Spot Hidden)" or "(Listen)" or "(Lockpick)").
- Do NOT output any TOOL_CALL blocks.
- Prefer actions that improve play flow: clear intent + clear target + plausible skill hint when appropriate.

State:
- Location: {loc}
- Alert: {alert}
- HP: {hp}
- SAN: {san}
""".strip()

        user = f"""{rules_arg}
[KEEPER_LAST]
{keeper_last.strip()}
[/KEEPER_LAST]

Turn {turn_index}: What do you do next?
""".strip()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = (self.client._chat(messages) or "").strip()

        # Defensive cleanup
        if "<TOOL_CALL>" in text:
            text = text.split("<TOOL_CALL>")[0].strip()
        if not text:
            text = "I take a steady breath and scan for anything out of place (Spot Hidden)."
        return text


# ----------------------------
# Helpers
# ----------------------------


def reset_game_state() -> None:
    save_state(DEFAULT_STATE)


# Rooms available in the (already-known) Corbitt House map - Ground Floor only.
# Keep names user-friendly; matching is case-insensitive.
START_ROOMS: List[str] = [
    # Ground floor (ROOM 1-6)
    "Storage Room",           # ROOM 1: A Storage Room
    "Second Storage Room",    # ROOM 2: A Second Storage Room
    "Mud Room",               # ROOM 3: The Mud Room
    "Living Room",            # ROOM 4: The Living Room
    "Dining Room",            # ROOM 5: The Dining Room
    "Kitchen",                # ROOM 6: Kitchen
]


def _pick_start_room(raw: str) -> str:
    """Pick a start room from START_ROOMS (case-insensitive)."""
    if not raw:
        return "Storage Room"
    r = raw.strip()
    if r.lower() == "random":
        import random

        return random.choice(START_ROOMS)
    # exact / case-insensitive match
    for name in START_ROOMS:
        if name.lower() == r.lower():
            return name
    # small alias support
    aliases = {
        "storage": "Storage Room",
        "storage room": "Storage Room",
        "storage 1": "Storage Room",
        "room 1": "Storage Room",
        "second storage": "Second Storage Room",
        "second storage room": "Second Storage Room",
        "storage 2": "Second Storage Room",
        "room 2": "Second Storage Room",
        "mud room": "Mud Room",
        "mud": "Mud Room",
        "room 3": "Mud Room",
        "living": "Living Room",
        "living room": "Living Room",
        "room 4": "Living Room",
        "dining": "Dining Room",
        "dining room": "Dining Room",
        "room 5": "Dining Room",
        "kitchen": "Kitchen",
        "room 6": "Kitchen",
    }
    if r.lower() in aliases:
        return aliases[r.lower()]
    print(f"[WARN] Unknown start room: {raw!r}. Falling back to 'Storage Room'.")
    print("[INFO] Available rooms:", ", ".join(START_ROOMS))
    return "Storage Room"


def set_start_room(room: str) -> None:
    """Set initial location in the saved state."""
    st = load_state()
    st.setdefault("game", {})
    st["game"]["location"] = room
    save_state(st)


def dump_state(tag: str) -> None:
    st = load_state()
    inv = st.get("investigator", {}) or {}
    game = st.get("game", {}) or {}
    print("\n" + "=" * 72)
    print(f"STATE {tag}")
    print("=" * 72)
    print(f"Location: {game.get('location')}")
    print(f"Turn: {game.get('turn_count')}")
    print(f"Alert: {game.get('alert_level')}")
    print(f"HP: {inv.get('hp')}")
    print(f"SAN: {inv.get('sanity')}")
    print(f"Inventory: {inv.get('inventory')}")
    print(f"Clues: {game.get('clues_found')}")
    print(f"Flags: {game.get('flags')}")
    print("=" * 72)


def build_keeper_rules_arg(
    engine,
    inv_msg: str,
    state: Dict[str, Any],
    *,
    search_type: str,
    return_raw_docs: bool,
) -> str:
    """Rules-only ARG for Keeper to improve check/effect correctness (still no spoilers)."""
    loc = (state.get("game", {}) or {}).get("location", "")
    retriever = (
        "RULES ONLY (no spoilers). "
        "Given the investigator action below, retrieve the relevant rules for: "
        "which skill/stat check applies, difficulty, how to resolve success/failure, "
        "and safe effect tokens (hp/sanity/alert/move/flag/clue/item) without inventing scenario content.\n\n"
        f"Location: {loc}\n"
        f"Investigator action: {inv_msg}"
    )
    docs = engine.retrieve(
        retriever,
        category=None,
        use_rerank=True,
        search_type=search_type,
        return_raw_docs=return_raw_docs,
    )
    # format small
    parts: List[str] = []
    for d in (docs or [])[:4]:
        if isinstance(d, str):
            parts.append(d.strip())
        else:
            parts.append(getattr(d, "page_content", "").strip())
    snippet = "\n".join([p for p in parts if p]).strip()
    if not snippet:
        return ""
    return f"[ARG]\n{snippet}\n[/ARG]\n"


# ----------------------------
# Main playtest
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Two-agent CoC playtest (rules-only RAG, no spoilers)"
    )
    parser.add_argument(
        "--start-room",
        default=os.getenv("START_ROOM", "Storage Room"),
        help=(
            "Starting room name (case-insensitive). Examples: 'Storage Room', 'Kitchen', 'Living Room', 'Mud Room'. "
            "Use 'random' to pick a random room. You can also set env START_ROOM."
        ),
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset state to DEFAULT_STATE before starting (keeps prior state).",
    )
    args = parser.parse_args()

    # Build/load DB once
    build_vector_database(reset=False)
    engine = get_engine()

    # Reset to default state (recommended for repeatable playtests)
    if not args.no_reset:
        reset_game_state()

    # Set start room (supports any already-known room on the map)
    start_room = _pick_start_room(args.start_room)
    st0 = load_state()
    st0.setdefault("game", {})
    st0["game"]["location"] = start_room
    save_state(st0)
    dump_state("(initial)")

    # Keeper agent (tools enabled)
    keeper = KeeperLLMClient(
        KeeperConfig(
            enable_rag=False,  # we inject rules-only ARG ourselves (keeps it deterministic and spoiler-safe)
            temperature=0.8,
            max_tokens=1200,
        )
    )

    # Investigator agent (same model backend via keeper._chat)
    investigator = InvestigatorAgent(
        chat_client=keeper,
        engine=engine,
        cfg=InvestigatorConfig(temperature=0.8, max_tokens=450),
    )

    # Playtest settings
    search_type = os.getenv(
        "RAG_SEARCH_TYPE", "mmr"
    )  # mmr/similarity/similarity_score_threshold
    return_raw_docs = False

    # Opening narration depends on start room (spoiler-safe)
    if start_room.lower() == "storage room":
        keeper_last = "You push the front door inward and enter the storage room. The Corbitt House exhales cold, stale air into your face."
    else:
        keeper_last = (
            f"You find yourself in the {start_room}. The house is quiet—too quiet. "
            "Dust hangs in the air, and every small sound seems to travel farther than it should."
        )
    print("\n=== KEEPER (opening) ===")
    print(keeper_last)

    # Run N turns
    N_TURNS = int(os.getenv("PLAYTEST_TURNS", "20"))
    for turn in range(1, N_TURNS + 1):
        st_before = load_state()

        inv_msg = investigator.propose_action(
            state=st_before,
            keeper_last=keeper_last,
            turn_index=turn,
            search_type=search_type,
            return_raw_docs=return_raw_docs,
        )
        print("\n=== INVESTIGATOR ===")
        print(inv_msg)

        # Rules-only ARG for Keeper (improves tool usage, still spoiler-safe)
        keeper_arg = build_keeper_rules_arg(
            engine,
            inv_msg,
            st_before,
            search_type=search_type,
            return_raw_docs=return_raw_docs,
        )

        keeper_out = keeper.run_turn((keeper_arg + inv_msg).strip())
        keeper_last = keeper_out

        print("\n=== KEEPER ===")
        print(keeper_out)

        dump_state(f"(after turn {turn})")
        time.sleep(0.05)

    print("\nPlaytest finished.")


if __name__ == "__main__":
    main()
