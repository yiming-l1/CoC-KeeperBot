"""
Keeper ↔ Investigator interaction demo for "The Haunting" (Corbitt House),

This demo is designed to exercise as many code paths as possible in:
  - resolve_action()
  - apply_effect_token()

Covered effect tokens:y
  - clue:...
  - item:...
  - sanity:0/1d6 , sanity:1/1d4
  - hp:-2
  - alert:+1
  - move:...
  - flag:...=true

Each turn:
- Investigator gives a natural-language action.
- Keeper (LLM) is explicitly instructed to emit exactly ONE <TOOL_CALL>
  invoking resolve_action().
- After each turn, the current game state is printed for verification.
"""

import os
import sys
import random
from src.llm_client import KeeperLLMClient, KeeperConfig
from src.state_manager import reset_state, load_state
from src.tools import print_state


def run_turn(client: KeeperLLMClient, title: str, investigator_msg: str):
    print("\n==============================")
    print("TURN:", title)
    print("=== Investigator ===")
    print(investigator_msg)

    output = client.run_turn(investigator_msg, force_tool=True)

    print("\n=== Keeper ===")
    print(output)

    print_state(title)


def main():
    # Make mechanics a bit more reproducible (if mechanics.py uses random)
    reset_state()
    random.seed(7)

    cfg = KeeperConfig(
        enable_rag=True,  # use LangChain RAG
        temperature=0.6,
        max_tokens=100,
        rag_k=6,
        rag_search_type="mmr",
    )
    client = KeeperLLMClient(cfg)

    # ------------------------------------------------------------
    # Turn 0: Enter the house (force MOVE token)
    # Coverage: move:
    # ------------------------------------------------------------
    run_turn(
        client,
        "T0 Enter Corbitt House (MOVE)",
        """
Start the game: I push the front door open and step into Corbitt House.
As the Keeper, you must first introduce the scene with atmosphere,
then advance the game state using exactly ONE tool call.

STRICT REQUIREMENTS:
- You must emit a <TOOL_CALL> invoking resolve_action.
- In stakes.effects, you MUST include: move:Storage Room
- Do NOT modify state directly in narration; movement must come from TOOL_RESULT.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 1: Search storage room (Spot Hidden)
    # Coverage: proposed_check (skill) + clue / alert
    # ------------------------------------------------------------
    run_turn(
        client,
        "T1 Storage Room Search (CHECK + CLUE / ALERT)",
        """
I enter the first-floor storage room and search carefully for anything unusual.

You must decide that a Spot Hidden (regular) check is required and then:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Spot Hidden / regular
- stakes.on_success MUST include: clue:boarded_cupboard
- stakes.on_failure MUST include: alert:+1

After TOOL_RESULT:
- On success: clearly describe the cupboard being boarded shut.
- On failure: describe making noise or increasing tension.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 2: Open the cupboard (get diaries)
    # Coverage: effects -> clue + item
    # ------------------------------------------------------------
    run_turn(
        client,
        "T2 Open the Cupboard (EFFECTS: CLUE + ITEM)",
        """
I pry open the boarded cupboard.

No skill check is required.
You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects BOTH:
  1) clue:Corbitt_Diaries
  2) item:Corbitt Diaries (3 volumes)

After TOOL_RESULT, narrate discovering the three diaries,
but do NOT reveal their full contents yet.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 3: Go upstairs (move to spare bedroom)
    # Coverage: move:
    # ------------------------------------------------------------
    run_turn(
        client,
        "T3 Go to Second Storage Room (MOVE)",
        """
I walk toward the second storage room.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: move:Second Storage Room

After TOOL_RESULT, describe the room:
the layout, the smell, and any notable details,
then offer exploration options.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 4: Witness supernatural event (SAN 1/1d4)
    # Coverage: sanity:1/1d4
    # ------------------------------------------------------------
    run_turn(
        client,
        "T4 Witness the Supernatural (SANITY 1/1d4)",
        """
I move closer and clearly witness something impossible
(e.g., furniture shifting on its own or the air distorting unnaturally).

You must NOT decide sanity loss yourself.
Instead:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: sanity:1/1d4

After TOOL_RESULT, narrate the shock and summarize the SAN change.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 5: Dodge an attack (HP loss on failure)
    # Coverage: Dodge check + hp:-2
    # ------------------------------------------------------------
    run_turn(
        client,
        "T5 Dodge a Violent Attack (CHECK + HP:-2)",
        """
A piece of furniture suddenly lunges toward me. I attempt to dodge out of the way.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Dodge / regular
- stakes.on_failure MUST include: hp:-2
- stakes.on_success may simply describe avoiding the attack

After TOOL_RESULT, narrate the outcome and summarize HP changes.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 6: Make noise (alert escalation)
    # Coverage: alert:+1
    # ------------------------------------------------------------
    run_turn(
        client,
        "T6 Loud Noise (ALERT:+1)",
        """
I shout down the hallway and kick a door, deliberately making a lot of noise.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: alert:+1

After TOOL_RESULT, describe how the house reacts to the disturbance.
Do NOT apply SAN or HP changes unless they come from the tool.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 7: Read disturbing symbols (SAN 0/1d6 with CHECK)
    # Coverage: sanity:0/1d6 with success/failure split
    # ------------------------------------------------------------
    run_turn(
        client,
        "T7 Disturbing Symbols (SANITY 0/1d6 + CHECK)",
        """
I briefly skim a page of the diaries covered in disturbing symbols,
without studying them in depth.

You must:
- Decide this requires a check using stat:POW (regular)
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = stat / POW / regular
- stakes.on_success MUST include: sanity:0/1d6
- stakes.on_failure MUST include: sanity:0/1d6

(Explanation: resolve_action will choose 0 or 1d6 based on success or failure.)

After TOOL_RESULT, narrate and summarize current SAN.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 8: Set a game flag
    # Coverage: flag:
    # ------------------------------------------------------------
    run_turn(
        client,
        "T8 Corbitt Influence Grows (FLAG)",
        """
While standing in the storage room, I feel a constant oppressive presence,
as if the house itself is watching me.

For demo purposes, you must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: flag:corbitt_influence=true

After TOOL_RESULT, narrate how this presence affects future exploration
without directly modifying SAN or HP.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 9: Severe SAN shock -> temp insanity
    # Coverage: sanity:1/1d10 (force large loss)
    # ------------------------------------------------------------
    run_turn(
        client,
        "T9 Severe Sanity Shock (TEMP INSANITY)",
        """
The realization of Corbitt’s rituals suddenly clicks together in my mind.
The implications are horrifying beyond reason.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: sanity:1/1d10

(This large loss is likely to trigger temp_insanity.)

After TOOL_RESULT:
- Narrate the onset of mental instability.
- Explicitly summarize current SAN and whether temp insanity is active.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 10: Night falls (FLAG)
    # Coverage: flag:night=true
    # ------------------------------------------------------------
    run_turn(
        client,
        "T10 Night Falls (FLAG: night)",
        """
Time passes, and night fully settles over Corbitt House.

For demo purposes:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: flag:night=true

After TOOL_RESULT:
- Describe how darkness changes the mood of the house.
- Do NOT apply SAN or HP changes unless from the tool.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 11: Hostile NPC appears
    # Coverage: flag:npc_hostile=true
    # ------------------------------------------------------------
    run_turn(
        client,
        "T11 NPC Turns Hostile (FLAG)",
        """
A gaunt, hostile figure steps out of the shadows at the edge of the room,
blocking the doorway.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:npc_hostile=true

After TOOL_RESULT:
- Narrate the NPC’s hostile posture and intent.
- Make it clear combat is now possible.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 11: Hostile NPC appears
    # Coverage: flag:npc_hostile=true
    # ------------------------------------------------------------
    run_turn(
        client,
        "T11 NPC Turns Hostile (FLAG)",
        """
A gaunt, hostile figure steps out of the shadows at the edge of the room,
blocking the doorway.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:npc_hostile=true

After TOOL_RESULT:
- Narrate the NPC’s hostile posture and intent.
- Make it clear combat is now possible.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 12: Enter combat mode
    # Coverage: flag:combat=true
    # ------------------------------------------------------------
    run_turn(
        client,
        "T12 Combat Begins (FLAG: combat)",
        """
The hostile figure suddenly lunges forward, forcing a fight.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:combat=true

After TOOL_RESULT:
- Narrate the transition into active combat.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 13: Combat Dodge (HP loss)
    # Coverage: Dodge check + hp:-2 (combat context)
    # ------------------------------------------------------------
    run_turn(
        client,
        "T13 Combat Dodge (CHECK + HP:-2)",
        """
The hostile NPC swings violently at me in the darkness.
I try to Dodge out of the way.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Dodge / regular
- stakes.on_failure MUST include: hp:-2

After TOOL_RESULT:
- Narrate the combat exchange.
- Summarize HP changes.
""".strip(),
    )

    # ------------------------------------------------------------
    # Turn 14: Door unlocked (escape route)
    # Coverage: flag:front_door_unlocked=true
    # ------------------------------------------------------------
    run_turn(
        client,
        "T14 Door Unlocked (FLAG)",
        """
Amid the chaos, I notice the front door is no longer locked.

For demo purposes:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:front_door_unlocked=true

After TOOL_RESULT:
- Narrate the possibility of escape or repositioning.
""".strip(),
    )

    print(
        "\n✅ Demo complete. You should see MOVE / CLUE / ITEM / SAN / HP / ALERT / FLAG effects reflected in state."
    )


if __name__ == "__main__":
    main()
