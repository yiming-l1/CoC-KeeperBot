import random
import re


def roll_d100():
    """Returns a random number between 1 and 100."""
    return random.randint(1, 100)


def check_skill_success(skill_level, difficulty="regular"):
    """
    Rolls d100 and compares against skill level.
    Returns: (roll_result, success_level_string)
    """
    roll = roll_d100()

    # Fumble Rule: If skill < 50, 96+ is fumble. If skill >= 50, 100 is fumble.
    is_fumble = (roll >= 96 and skill_level < 50) or (roll == 100 and skill_level >= 50)
    if is_fumble:
        return roll, "Fumble"

    # Critical Success (01)
    if roll == 1:
        return roll, "Critical Success"

    # Success Levels
    hard_target = skill_level // 2
    extreme_target = skill_level // 5

    if roll <= extreme_target:
        return roll, "Extreme Success"
    elif roll <= hard_target:
        return roll, "Hard Success"
    elif roll <= skill_level:
        return roll, "Regular Success"
    else:
        return roll, "Failure"


def calculate_sanity_loss(current_san: int, loss_str: str):
    """
    Supports:
      - "0", "5"
      - "1d4", "2d6"
      - "d6" (treated as "1d6")
    Returns: (new_san, loss_val, temp_insanity)
    """
    s = str(loss_str).strip().lower()

    if s in ("0", ""):
        loss_val = 0
    else:
        # allow "d6" as "1d6"
        m = re.fullmatch(r"(\d*)d(\d+)", s)
        if m:
            num_s, die_s = m.groups()
            num = int(num_s) if num_s not in (None, "") else 1
            die = int(die_s)
            loss_val = sum(random.randint(1, die) for _ in range(num))
        else:
            loss_val = int(s)  # will raise ValueError if invalid

    new_san = max(0, int(current_san) - loss_val)

    # simplified rule: one-hit loss >= 5 => temp insanity
    temp_insanity = loss_val >= 5

    return new_san, loss_val, temp_insanity
