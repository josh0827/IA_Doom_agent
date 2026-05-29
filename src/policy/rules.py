from src.policy.actions import Action

ENEMIGOS = {"imp", "demon", "cacodemon"}
RECURSOS = {"medkit", "ammo", "armor"}


def decidir(result, vida: int, ammo: int, frame_w: int) -> Action:
    if result is None or len(result.boxes) == 0:
        return Action.TURN_RIGHT

    centro_x = frame_w / 2
    enemigos = []
    items = []

    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        x1, _, x2, _ = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        if cls_name in ENEMIGOS:
            enemigos.append((cls_name, cx))
        elif cls_name in RECURSOS:
            items.append((cls_name, cx))

    if vida < 30 and any(it[0] == "medkit" for it in items):
        target = next(it for it in items if it[0] == "medkit")
        return _girar_hacia(target[1], centro_x)

    if enemigos and ammo > 0:
        objetivo = min(enemigos, key=lambda e: abs(e[1] - centro_x))
        if abs(objetivo[1] - centro_x) < 40:
            return Action.ATTACK
        return _girar_hacia(objetivo[1], centro_x)

    return Action.MOVE_FORWARD


def _girar_hacia(target_x: float, centro_x: float) -> Action:
    return Action.TURN_LEFT if target_x < centro_x else Action.TURN_RIGHT
