import json
import os
import copy

STATE_FILE = "./data/game_state.json"

DEFAULT_STATE = {
    "investigator": {
        "name": "Detective Skywalker",
        "stats": {
            "STR": 60,
            "CON": 50,
            "SIZ": 60,
            "DEX": 50,
            "APP": 40,
            "INT": 70,
            "POW": 40,
            "EDU": 70,
        },
        "hp": {"current": 11, "max": 11},
        "sanity": {"current": 40, "max": 99, "temp_insanity": False},
        "skills": {
            "Spot Hidden": 55,
            "Library Use": 70,
            "Fighting (Brawl)": 40,
            "Dodge": 30,
            "History": 40,
            "Listen": 50,
            "Stealth": 40,
            "Persuade": 80,
            "Psychology": 50,
            "First Aid": 30,
            "Lockpick": 76,
        },
        "inventory": ["Flashlight", "Notebook", "Pencil"],
    },
    "game": {"location": "Foyer", "clues_found": [], "turn_count": 0, "alert_level": 0},
}


def load_state():
    if not os.path.exists(STATE_FILE):
        save_state(DEFAULT_STATE)
        return copy.deepcopy(DEFAULT_STATE)
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    """Writes the state back to the JSON file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def reset_state():
    save_state(DEFAULT_STATE)
    return load_state()


def update_sanity(amount):
    """Updates Sanity and auto-saves."""
    state = load_state()
    # amount is negative for loss
    state["investigator"]["sanity"]["current"] += amount

    # Simple check: If SAN drops too low, flag insanity
    if amount <= -5:
        state["investigator"]["sanity"]["temp_insanity"] = True

    save_state(state)
    return state["investigator"]["sanity"]["current"]


def add_inventory_item(item_name):
    """Adds an item to the list."""
    state = load_state()
    if item_name not in state["investigator"]["inventory"]:
        state["investigator"]["inventory"].append(item_name)
        save_state(state)
        return True
    return False


def add_clue(clue_id: str) -> bool:
    state = load_state()
    clues = state["game"].setdefault("clues_found", [])
    if clue_id not in clues:
        clues.append(clue_id)
        save_state(state)
        return True
    return False


def update_hp(delta: int) -> int:
    """
    Updates HP by delta (negative for damage, positive for healing),
    clamps to [0, max], auto-saves, returns new current HP.
    """
    state = load_state()
    hp = state["investigator"].setdefault("hp", {"current": 0, "max": 0})
    cur = int(hp.get("current", 0))
    max_hp = int(hp.get("max", cur))
    new_cur = max(0, min(max_hp, cur + int(delta)))
    hp["current"] = new_cur
    save_state(state)
    return new_cur


def set_location(location: str) -> str:
    """
    Sets game location, auto-saves, returns the location.
    """
    state = load_state()
    game = state.setdefault("game", {})
    game["location"] = str(location)
    save_state(state)
    return game["location"]


def set_flag(key: str, value):
    """
    Sets a boolean/string flag under state["game"][key], auto-saves.
    """
    state = load_state()
    game = state.setdefault("game", {})
    game[str(key)] = value
    save_state(state)


def increment_turn() -> int:
    """
    Increments game.turn_count by 1, auto-saves, returns new turn_count.
    """
    state = load_state()
    game = state.setdefault("game", {})
    game["turn_count"] = int(game.get("turn_count", 0)) + 1
    save_state(state)
    return game["turn_count"]
