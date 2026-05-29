from enum import IntEnum


class Action(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ATTACK = 4
    USE = 5
    IDLE = 6


def action_to_vizdoom(action: Action) -> list[int]:
    vector = [0] * 6
    if action == Action.MOVE_FORWARD:
        vector[0] = 1
    elif action == Action.MOVE_BACKWARD:
        vector[1] = 1
    elif action == Action.TURN_LEFT:
        vector[2] = 1
    elif action == Action.TURN_RIGHT:
        vector[3] = 1
    elif action == Action.ATTACK:
        vector[4] = 1
    elif action == Action.USE:
        vector[5] = 1
    return vector
