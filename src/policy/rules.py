from src.policy.actions import Action

ENEMIGOS = {
    "imp", "cacodemon", "baron-of-hell", "cyberdemon",
    "lost-soul", "pinky", "shotgun-guy", "specter",
    "spiderdemon", "zombieman",
}


def decidir(result, vida: int, ammo: int, frame_w: int) -> Action:
    if result is None or len(result.boxes) == 0:
        return Action.MOVE_FORWARD  # sin enemigos: avanzar por el pasillo

    centro_x = frame_w / 2
    enemigos = []

    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        x1, _, x2, _ = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        if cls_name in ENEMIGOS:
            enemigos.append((cls_name, cx))

    if enemigos and ammo > 0:
        objetivo = min(enemigos, key=lambda e: abs(e[1] - centro_x))
        distancia_x = abs(objetivo[1] - centro_x)
        if distancia_x < frame_w * 0.25:  # 25% del ancho = zona de disparo amplia
            return Action.ATTACK
        return _girar_hacia(objetivo[1], centro_x)

    return Action.MOVE_FORWARD


def _girar_hacia(target_x: float, centro_x: float) -> Action:
    return Action.TURN_LEFT if target_x < centro_x else Action.TURN_RIGHT
