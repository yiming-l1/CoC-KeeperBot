#!/usr/bin/env python3
"""
Analyze playtest logs to extract metrics about tool calls, skill checks, and game mechanics.

Metrics extracted:
- Total number of turns
- Number of tool calls (TOOL_CALL)
- Number of skill checks performed
- Number of no_check turns (no dice roll)
- Types of effects applied (clue, item, sanity, hp, move, flag, alert)
- Average tool calls per turn
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


def analyze_log_file(log_path: Path) -> Optional[Dict[str, Any]]:
    """Analyze a single log file and extract metrics."""
    if not log_path.exists():
        return None

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {
        "file": str(log_path.name),
        "total_turns": 0,
        "tool_calls": 0,
        "skill_checks": 0,
        "no_check_turns": 0,
        "effects_applied": defaultdict(int),
        "effect_types": defaultdict(int),
        "sanity_changes": 0,
        "hp_changes": 0,
        "moves": 0,
        "clues_found": 0,
        "items_found": 0,
    }

    # Count turns
    turn_pattern = r"TURN:\s*(T\d+\s+[^\n]+)"
    turns = re.findall(turn_pattern, content)
    metrics["total_turns"] = len(turns)

    # Count tool calls
    tool_call_pattern = r'\[DEBUG\] chat respond-TOOL_CALL:'
    tool_calls = re.findall(tool_call_pattern, content)
    metrics["tool_calls"] = len(tool_calls)

    # Count skill checks (checks that actually rolled dice)
    # Match both single and double quotes: {'type': 'skill'} or {"type": "skill"}
    # Also match 'type': or "type": patterns
    check_pattern = r"\[DEBUG\] CHECK:\s*\{[^}]*['\"]type['\"]\s*:\s*['\"](?:skill|stat)['\"]"
    checks = re.findall(check_pattern, content)
    metrics["skill_checks"] = len(checks)

    # Count no_check turns
    no_check_pattern = r'\[DEBUG\] CHECK:\s*no_check'
    no_checks = re.findall(no_check_pattern, content)
    metrics["no_check_turns"] = len(no_checks)

    # Count effects applied
    applied_patterns = {
        "clue": r'\[DEBUG\] Applied clue:',
        "item": r'\[DEBUG\] Applied item:',
        "sanity": r'\[DEBUG\] Applied sanity',
        "hp": r'\[DEBUG\] Applied hp',
        "move": r'\[DEBUG\] Applied move to:',
        "alert": r'\[DEBUG\] Applied alert',
        "flag": r'\[DEBUG\] Applied flag',
    }

    for effect_type, pattern in applied_patterns.items():
        matches = re.findall(pattern, content)
        count = len(matches)
        metrics["effects_applied"][effect_type] = count
        if count > 0:
            metrics["effect_types"][effect_type] = count

    # Specific counts
    metrics["sanity_changes"] = metrics["effects_applied"]["sanity"]
    metrics["hp_changes"] = metrics["effects_applied"]["hp"]
    metrics["moves"] = metrics["effects_applied"]["move"]
    metrics["clues_found"] = metrics["effects_applied"]["clue"]
    metrics["items_found"] = metrics["effects_applied"]["item"]

    # Calculate averages
    if metrics["total_turns"] > 0:
        metrics["tool_calls_per_turn"] = metrics["tool_calls"] / metrics["total_turns"]
        metrics["skill_checks_per_turn"] = (
            metrics["skill_checks"] / metrics["total_turns"]
        )
        metrics["no_check_rate"] = (
            metrics["no_check_turns"] / metrics["total_turns"]
        )
    else:
        metrics["tool_calls_per_turn"] = 0.0
        metrics["skill_checks_per_turn"] = 0.0
        metrics["no_check_rate"] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Print metrics in a readable format."""
    print(f"\n{'='*72}")
    print(f"LOG ANALYSIS: {metrics['file']}")
    print(f"{'='*72}\n")

    print(f"Total Turns: {metrics['total_turns']}")
    print(f"Tool Calls: {metrics['tool_calls']}")
    print(f"  Average per turn: {metrics['tool_calls_per_turn']:.2f}")
    print()

    print(f"Skill Checks (dice rolls): {metrics['skill_checks']}")
    print(f"  Average per turn: {metrics['skill_checks_per_turn']:.2f}")
    print(f"No-Check Turns: {metrics['no_check_turns']}")
    print(f"  Rate: {metrics['no_check_rate']:.1%}")
    print()

    print("Effects Applied:")
    for effect_type, count in sorted(metrics["effect_types"].items()):
        print(f"  - {effect_type}: {count}")
    print()

    print("Game Mechanics Summary:")
    print(f"  - Moves: {metrics['moves']}")
    print(f"  - Clues found: {metrics['clues_found']}")
    print(f"  - Items found: {metrics['items_found']}")
    print(f"  - Sanity changes: {metrics['sanity_changes']}")
    print(f"  - HP changes: {metrics['hp_changes']}")
    print()


def compare_logs(log_metrics: List[Dict[str, Any]]):
    """Compare metrics across multiple log files."""
    if len(log_metrics) == 0:
        return

    print(f"\n{'='*72}")
    print("COMPARISON ACROSS LOG FILES")
    print(f"{'='*72}\n")

    # Aggregate metrics
    total_turns = sum(m["total_turns"] for m in log_metrics)
    total_tool_calls = sum(m["tool_calls"] for m in log_metrics)
    total_skill_checks = sum(m["skill_checks"] for m in log_metrics)
    total_no_checks = sum(m["no_check_turns"] for m in log_metrics)

    if total_turns > 0:
        avg_tool_calls_per_turn = total_tool_calls / total_turns
        avg_skill_checks_per_turn = total_skill_checks / total_turns
        no_check_rate = total_no_checks / total_turns
    else:
        avg_tool_calls_per_turn = 0.0
        avg_skill_checks_per_turn = 0.0
        no_check_rate = 0.0

    print(f"Total Logs Analyzed: {len(log_metrics)}")
    print(f"Total Turns: {total_turns}")
    print(f"Total Tool Calls: {total_tool_calls}")
    print(f"  Average per turn: {avg_tool_calls_per_turn:.2f}")
    print(f"Total Skill Checks: {total_skill_checks}")
    print(f"  Average per turn: {avg_skill_checks_per_turn:.2f}")
    print(f"No-Check Rate: {no_check_rate:.1%}")
    print()

    # Aggregate effects
    all_effects: Dict[str, int] = defaultdict(int)
    for metrics in log_metrics:
        for effect_type, count in metrics["effect_types"].items():
            all_effects[effect_type] += count

    if all_effects:
        print("Total Effects Applied:")
        for effect_type, count in sorted(all_effects.items()):
            print(f"  - {effect_type}: {count}")
        print()


def main():
    """Main function to analyze log files."""
    log_dir = Path("./log")

    if not log_dir.exists():
        print(f"Error: {log_dir} directory not found")
        sys.exit(1)

    # Get log files from command line or analyze all
    if len(sys.argv) > 1:
        log_files = [Path(f) for f in sys.argv[1:]]
        # If relative paths, prepend log_dir
        log_files = [
            (log_dir / f.name if not f.is_absolute() else f) for f in log_files
        ]
    else:
        # Analyze all log files
        log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print(f"No log files found in {log_dir}")
        sys.exit(1)

    print(f"Analyzing {len(log_files)} log file(s)...\n")

    all_metrics = []
    for log_file in sorted(log_files):
        metrics = analyze_log_file(log_file)
        if metrics:
            print_metrics(metrics)
            all_metrics.append(metrics)

    # Compare if multiple files
    if len(all_metrics) > 1:
        compare_logs(all_metrics)


if __name__ == "__main__":
    main()

