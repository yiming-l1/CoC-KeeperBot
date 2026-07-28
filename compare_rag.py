#!/usr/bin/env python3
"""
RAG vs No-RAG Comparison Script

Uses fixed test cases from test_keeper_investigator_inter.py to compare
Keeper responses with and without RAG.

Usage:
    python compare_rag.py
"""

from __future__ import annotations

import os
import json
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.llm_client import KeeperLLMClient, KeeperConfig
from src.state_manager import reset_state, load_state, save_state, DEFAULT_STATE


# ----------------------------
# Fixed Test Cases (from test_keeper_investigator_inter.py)
# ----------------------------

TEST_CASES = [
    {
        "turn": 0,
        "title": "T0 Enter Corbitt House (MOVE)",
        "investigator_msg": """Start the game: I push the front door open and step into Corbitt House.
As the Keeper, you must first introduce the scene with atmosphere,
then advance the game state using exactly ONE tool call.

STRICT REQUIREMENTS:
- You must emit a <TOOL_CALL> invoking resolve_action.
- In stakes.effects, you MUST include: move:Storage Room
- Do NOT modify state directly in narration; movement must come from TOOL_RESULT.""",
    },
    {
        "turn": 1,
        "title": "T1 Storage Room Search (CHECK + CLUE / ALERT)",
        "investigator_msg": """I enter the first-floor storage room and search carefully for anything unusual.

You must decide that a Spot Hidden (regular) check is required and then:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Spot Hidden / regular
- stakes.on_success MUST include: clue:boarded_cupboard
- stakes.on_failure MUST include: alert:+1

After TOOL_RESULT:
- On success: clearly describe the cupboard being boarded shut.
- On failure: describe making noise or increasing tension.""",
    },
    {
        "turn": 2,
        "title": "T2 Open the Cupboard (EFFECTS: CLUE + ITEM)",
        "investigator_msg": """I pry open the boarded cupboard.

No skill check is required.
You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects BOTH:
  1) clue:Corbitt_Diaries
  2) item:Corbitt Diaries (3 volumes)

After TOOL_RESULT, narrate discovering the three diaries,
but do NOT reveal their full contents yet.""",
    },
    {
        "turn": 3,
        "title": "T3 Go to Second Storage Room (MOVE)",
        "investigator_msg": """I walk toward the second storage room.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: move:Second Storage Room

After TOOL_RESULT, describe the room:
the layout, the smell, and any notable details,
then offer exploration options.""",
    },
    {
        "turn": 4,
        "title": "T4 Witness the Supernatural (SANITY 1/1d4)",
        "investigator_msg": """I move closer and clearly witness something impossible
(e.g., furniture shifting on its own or the air distorting unnaturally).

You must NOT decide sanity loss yourself.
Instead:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: sanity:1/1d4

After TOOL_RESULT, narrate the shock and summarize the SAN change.""",
    },
    {
        "turn": 5,
        "title": "T5 Dodge a Violent Attack (CHECK + HP:-2)",
        "investigator_msg": """A piece of furniture suddenly lunges toward me. I attempt to dodge out of the way.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Dodge / regular
- stakes.on_failure MUST include: hp:-2
- stakes.on_success may simply describe avoiding the attack

After TOOL_RESULT, narrate the outcome and summarize HP changes.""",
    },
    {
        "turn": 6,
        "title": "T6 Loud Noise (ALERT:+1)",
        "investigator_msg": """I shout down the hallway and kick a door, deliberately making a lot of noise.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: alert:+1

After TOOL_RESULT, describe how the house reacts to the disturbance.
Do NOT apply SAN or HP changes unless they come from the tool.""",
    },
    {
        "turn": 7,
        "title": "T7 Disturbing Symbols (SANITY 0/1d6 + CHECK)",
        "investigator_msg": """I briefly skim a page of the diaries covered in disturbing symbols,
without studying them in depth.

You must:
- Decide this requires a check using stat:POW (regular)
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = stat / POW / regular
- stakes.on_success MUST include: sanity:0/1d6
- stakes.on_failure MUST include: sanity:0/1d6

(Explanation: resolve_action will choose 0 or 1d6 based on success or failure.)

After TOOL_RESULT, narrate and summarize current SAN.""",
    },
    {
        "turn": 8,
        "title": "T8 Corbitt Influence Grows (FLAG)",
        "investigator_msg": """While standing in the storage room, I feel a constant oppressive presence,
as if the house itself is watching me.

For demo purposes, you must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: flag:corbitt_influence=true

After TOOL_RESULT, narrate how this presence affects future exploration
without directly modifying SAN or HP.""",
    },
    {
        "turn": 9,
        "title": "T9 Severe Sanity Shock (TEMP INSANITY)",
        "investigator_msg": """The realization of Corbitt's rituals suddenly clicks together in my mind.
The implications are horrifying beyond reason.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: sanity:1/1d10

(This large loss is likely to trigger temp_insanity.)

After TOOL_RESULT:
- Narrate the onset of mental instability.
- Explicitly summarize current SAN and whether temp insanity is active.""",
    },
    {
        "turn": 10,
        "title": "T10 Night Falls (FLAG: night)",
        "investigator_msg": """Time passes, and night fully settles over Corbitt House.

For demo purposes:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects: flag:night=true

After TOOL_RESULT:
- Describe how darkness changes the mood of the house.
- Do NOT apply SAN or HP changes unless from the tool.""",
    },
    {
        "turn": 11,
        "title": "T11 NPC Turns Hostile (FLAG)",
        "investigator_msg": """A gaunt, hostile figure steps out of the shadows at the edge of the room,
blocking the doorway.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:npc_hostile=true

After TOOL_RESULT:
- Narrate the NPC's hostile posture and intent.
- Make it clear combat is now possible.""",
    },
    {
        "turn": 12,
        "title": "T12 Combat Begins (FLAG: combat)",
        "investigator_msg": """The hostile figure suddenly lunges forward, forcing a fight.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:combat=true

After TOOL_RESULT:
- Narrate the transition into active combat.""",
    },
    {
        "turn": 13,
        "title": "T13 Combat Dodge (CHECK + HP:-2)",
        "investigator_msg": """The hostile NPC swings violently at me in the darkness.
I try to Dodge out of the way.

You must:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- proposed_check = skill / Dodge / regular
- stakes.on_failure MUST include: hp:-2

After TOOL_RESULT:
- Narrate the combat exchange.
- Summarize HP changes.""",
    },
    {
        "turn": 14,
        "title": "T14 Door Unlocked (FLAG)",
        "investigator_msg": """Amid the chaos, I notice the front door is no longer locked.

For demo purposes:
- Use exactly ONE <TOOL_CALL> invoking resolve_action
- Include in stakes.effects:
  - flag:front_door_unlocked=true

After TOOL_RESULT:
- Narrate the possibility of escape or repositioning.""",
    },
]


# ----------------------------
# Evaluation Metrics
# ----------------------------

@dataclass
class ObjectiveMetrics:
    turn: int
    has_spoiler: bool = False
    spoiler_details: List[str] = field(default_factory=list)
    has_rule_violation: bool = False
    rule_violations: List[str] = field(default_factory=list)
    has_hallucination: bool = False
    hallucination_entities: List[str] = field(default_factory=list)
    has_tool_call: bool = False
    tool_call_valid: bool = False


VALID_SKILLS = [
    "Spot Hidden", "Listen", "Library Use", "Fight Brawl", "Fight Melee",
    "Firearms (Handgun)", "Firearms (Rifle/Shotgun)", "Dodge", "Stealth",
    "Persuade", "Fast Talk", "Intimidate", "Charm", "Psychology",
    "Lockpick", "Mechanical Repair", "Electrical Repair", "First Aid",
    "Medicine", "Surgery", "Chemistry", "Biology", "Geology", "Physics",
    "Astronomy", "Archaeology", "History", "Natural World", "Occult",
    "Accountancy", "Law", "Language", "Pilot", "Drive Auto", "Ride",
    "Swim", "Climb", "Jump", "Throw", "Art/Craft", "Disguise", "Sleight of Hand",
]

FORBIDDEN_ENTITIES = [
    "cthulhu", "azathoth", "nyarlathotep", "yog-sothoth", "shoggoth",
    "deep ones", "deep one",
]


def extract_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """Extract tool call JSON from response if present."""
    tool_block_match = re.search(
        r"<TOOL_CALL>(.*?)</TOOL_CALL>",
        response,
        flags=re.DOTALL | re.IGNORECASE
    )
    if tool_block_match:
        try:
            return json.loads(tool_block_match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def evaluate_turn(response: str, state_before: Dict[str, Any]) -> ObjectiveMetrics:
    """Evaluate a single turn's objective metrics."""
    metrics = ObjectiveMetrics(turn=0)
    
    tool_call_json = extract_tool_call(response)
    metrics.has_tool_call = tool_call_json is not None
    
    if tool_call_json:
        if tool_call_json.get("name") == "resolve_action":
            args = tool_call_json.get("arguments", {})
            if "intent" in args and "scene_id" in args:
                metrics.tool_call_valid = True
                
                # Check skill names
                proposed_check = args.get("proposed_check")
                if proposed_check and isinstance(proposed_check, dict):
                    skill_name = proposed_check.get("name", "")
                    if skill_name:
                        skill_lower = skill_name.lower()
                        valid_skills_lower = [s.lower() for s in VALID_SKILLS]
                        if skill_lower not in valid_skills_lower:
                            common_errors = {
                                "perception": "Should use 'Spot Hidden' or 'Listen'",
                                "observation": "Should use 'Spot Hidden'",
                                "awareness": "Should use 'Spot Hidden'",
                                "investigation": "Should use 'Spot Hidden' or 'Library Use'",
                            }
                            if skill_lower in common_errors:
                                metrics.rule_violations.append(f"Invalid skill: '{skill_name}' - {common_errors[skill_lower]}")
                            else:
                                metrics.rule_violations.append(f"Unknown skill: '{skill_name}'")
            else:
                metrics.tool_call_valid = False
                metrics.rule_violations.append("Missing required fields (intent/scene_id)")
    
    # Check hallucinations
    response_lower = response.lower()
    for entity in FORBIDDEN_ENTITIES:
        if entity in response_lower:
            context = response_lower[max(0, response_lower.find(entity)-50):response_lower.find(entity)+100]
            negative_indicators = ["no", "not", "don't see", "can't find", "isn't", "doesn't", "nothing like"]
            if not any(neg in context for neg in negative_indicators):
                metrics.hallucination_entities.append(entity)
    
    if metrics.rule_violations:
        metrics.has_rule_violation = True
    if metrics.hallucination_entities:
        metrics.has_hallucination = True
    
    return metrics


# ----------------------------
# Main Comparison Function
# ----------------------------

def run_test_case(client: KeeperLLMClient, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single test case and return results."""
    state_before = load_state()
    investigator_msg = test_case["investigator_msg"].strip()
    
    response = client.run_turn(investigator_msg, force_tool=True)
    
    state_after = load_state()
    metrics = evaluate_turn(response, state_before)
    metrics.turn = test_case["turn"]
    
    return {
        "turn": test_case["turn"],
        "title": test_case["title"],
        "investigator_msg": investigator_msg,
        "keeper_response": response,
        "state_before": state_before,
        "state_after": state_after,
        "metrics": metrics,
    }


def run_all_tests(rag_enabled: bool) -> List[Dict[str, Any]]:
    """Run all test cases with RAG enabled or disabled."""
    reset_state()
    
    cfg = KeeperConfig(
        enable_rag=rag_enabled,
        temperature=0.6,
        max_tokens=1200,
        rag_k=6,
        rag_search_type="mmr",
    )
    client = KeeperLLMClient(cfg)
    
    results = []
    for test_case in TEST_CASES:
        print(f"\n[{'RAG' if rag_enabled else 'NO-RAG'}] Running {test_case['title']}...")
        result = run_test_case(client, test_case)
        results.append(result)
    
    return results


def generate_report(results_no_rag: List[Dict[str, Any]], results_with_rag: List[Dict[str, Any]], 
                    output_path: str) -> None:
    """Generate comparison report."""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("RAG vs NO-RAG COMPARISON REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Fixed test cases from test_keeper_investigator_inter.py")
    report_lines.append("")
    
    # Summary statistics
    def calc_summary(results):
        total = len(results)
        spoilers = sum(1 for r in results if r["metrics"].has_spoiler)
        rule_violations = sum(1 for r in results if r["metrics"].has_rule_violation)
        hallucinations = sum(1 for r in results if r["metrics"].has_hallucination)
        tool_calls = sum(1 for r in results if r["metrics"].has_tool_call)
        valid_tool_calls = sum(1 for r in results if r["metrics"].has_tool_call and r["metrics"].tool_call_valid)
        return {
            "total": total,
            "spoilers": spoilers,
            "rule_violations": rule_violations,
            "hallucinations": hallucinations,
            "tool_calls": tool_calls,
            "valid_tool_calls": valid_tool_calls,
        }
    
    summary_no_rag = calc_summary(results_no_rag)
    summary_with_rag = calc_summary(results_with_rag)
    
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 100)
    report_lines.append(f"NO-RAG:")
    report_lines.append(f"  Total Turns: {summary_no_rag['total']}")
    report_lines.append(f"  Rule Violations: {summary_no_rag['rule_violations']} ({summary_no_rag['rule_violations']/summary_no_rag['total']*100:.1f}%)")
    report_lines.append(f"  Hallucinations: {summary_no_rag['hallucinations']} ({summary_no_rag['hallucinations']/summary_no_rag['total']*100:.1f}%)")
    report_lines.append(f"  Tool Calls: {summary_no_rag['tool_calls']} ({summary_no_rag['valid_tool_calls']} valid)")
    report_lines.append("")
    report_lines.append(f"WITH-RAG:")
    report_lines.append(f"  Total Turns: {summary_with_rag['total']}")
    report_lines.append(f"  Rule Violations: {summary_with_rag['rule_violations']} ({summary_with_rag['rule_violations']/summary_with_rag['total']*100:.1f}%)")
    report_lines.append(f"  Hallucinations: {summary_with_rag['hallucinations']} ({summary_with_rag['hallucinations']/summary_with_rag['total']*100:.1f}%)")
    report_lines.append(f"  Tool Calls: {summary_with_rag['tool_calls']} ({summary_with_rag['valid_tool_calls']} valid)")
    report_lines.append("")
    
    # Turn-by-turn comparison
    report_lines.append("=" * 100)
    report_lines.append("TURN-BY-TURN COMPARISON")
    report_lines.append("=" * 100)
    
    for i in range(len(TEST_CASES)):
        turn_num = TEST_CASES[i]["turn"]
        title = TEST_CASES[i]["title"]
        
        report_lines.append("")
        report_lines.append("-" * 100)
        report_lines.append(f"TURN {turn_num}: {title}")
        report_lines.append("-" * 100)
        
        result_no_rag = results_no_rag[i]
        result_with_rag = results_with_rag[i]
        
        report_lines.append(f"\n[INVESTIGATOR ACTION]")
        report_lines.append(result_no_rag["investigator_msg"][:200] + ("..." if len(result_no_rag["investigator_msg"]) > 200 else ""))
        
        report_lines.append(f"\n[NO-RAG RESPONSE]")
        response_no_rag = result_no_rag["keeper_response"]
        report_lines.append(response_no_rag[:500] + ("..." if len(response_no_rag) > 500 else ""))
        report_lines.append("")
        m_no_rag = result_no_rag["metrics"]
        if m_no_rag.has_rule_violation:
            report_lines.append(f"  ❌ RULE VIOLATION: {', '.join(m_no_rag.rule_violations)}")
        if m_no_rag.has_hallucination:
            report_lines.append(f"  ⚠️  HALLUCINATION: {', '.join(m_no_rag.hallucination_entities)}")
        if m_no_rag.has_tool_call:
            report_lines.append(f"  {'✅' if m_no_rag.tool_call_valid else '❌'} TOOL CALL: {'Valid' if m_no_rag.tool_call_valid else 'Invalid'}")
        
        report_lines.append(f"\n[WITH-RAG RESPONSE]")
        response_with_rag = result_with_rag["keeper_response"]
        report_lines.append(response_with_rag[:500] + ("..." if len(response_with_rag) > 500 else ""))
        report_lines.append("")
        m_with_rag = result_with_rag["metrics"]
        if m_with_rag.has_rule_violation:
            report_lines.append(f"  ❌ RULE VIOLATION: {', '.join(m_with_rag.rule_violations)}")
        if m_with_rag.has_hallucination:
            report_lines.append(f"  ⚠️  HALLUCINATION: {', '.join(m_with_rag.hallucination_entities)}")
        if m_with_rag.has_tool_call:
            report_lines.append(f"  {'✅' if m_with_rag.tool_call_valid else '❌'} TOOL CALL: {'Valid' if m_with_rag.tool_call_valid else 'Invalid'}")
        
        report_lines.append("\n[HUMAN EVALUATION]")
        report_lines.append("Rate on:")
        report_lines.append("  1. Narrative Quality (1-5)")
        report_lines.append("  2. Rule Accuracy (1-5)")
        report_lines.append("  3. Consistency (1-5)")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"\n✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare RAG vs No-RAG using fixed test cases")
    parser.add_argument(
        "--output",
        default="./evaluation/rag_comparison_report.txt",
        help="Output path for comparison report",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("RAG vs NO-RAG COMPARISON")
    print("=" * 80)
    print(f"Running {len(TEST_CASES)} fixed test cases...")
    print()
    
    # Run NO-RAG tests
    print("[1/2] Running NO-RAG tests...")
    results_no_rag = run_all_tests(rag_enabled=False)
    print("✅ NO-RAG tests complete")
    
    # Run WITH-RAG tests
    print("\n[2/2] Running WITH-RAG tests...")
    results_with_rag = run_all_tests(rag_enabled=True)
    print("✅ WITH-RAG tests complete")
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_report(results_no_rag, results_with_rag, args.output)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

