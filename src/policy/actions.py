from enum import IntEnum


class Action(IntEnum):
    MOVE_FORWARD = 0
    MOVE_BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    ATTACK = 4
    STRAFE_LEFT = 5          # esquiva lateral izquierda
    STRAFE_RIGHT = 6         # esquiva lateral derecha
    FORWARD_ATTACK = 7       # avanzar + disparar simultaneamente
    STRAFE_LEFT_ATTACK = 8   # esquivar izquierda + disparar
    STRAFE_RIGHT_ATTACK = 9  # esquivar derecha + disparar
    TURN_LEFT_ATTACK = 10    # girar izquierda + disparar (apuntar mientras gira)
    TURN_RIGHT_ATTACK = 11   # girar derecha + disparar
    BACKWARD_ATTACK = 12     # retroceder + disparar (kiting: mantiene distancia)


# Vector: [FORWARD, BACKWARD, TURN_L, TURN_R, ATTACK, USE, MOVE_L, MOVE_R]
def action_to_vizdoom(action: Action) -> list[int]:
    v = [0] * 8
    if action == Action.MOVE_FORWARD:
        v[0] = 1
    elif action == Action.MOVE_BACKWARD:
        v[1] = 1
    elif action == Action.TURN_LEFT:
        v[2] = 1
    elif action == Action.TURN_RIGHT:
        v[3] = 1
    elif action == Action.ATTACK:
        v[4] = 1
    elif action == Action.STRAFE_LEFT:
        v[6] = 1
    elif action == Action.STRAFE_RIGHT:
        v[7] = 1
    elif action == Action.FORWARD_ATTACK:
        v[0] = 1; v[4] = 1
    elif action == Action.STRAFE_LEFT_ATTACK:
        v[6] = 1; v[4] = 1
    elif action == Action.STRAFE_RIGHT_ATTACK:
        v[7] = 1; v[4] = 1
    elif action == Action.TURN_LEFT_ATTACK:
        v[2] = 1; v[4] = 1
    elif action == Action.TURN_RIGHT_ATTACK:
        v[3] = 1; v[4] = 1
    elif action == Action.BACKWARD_ATTACK:
        v[1] = 1; v[4] = 1
    return v
