import random

from src.policy.actions import Action

ENEMIGOS = {
    "imp", "cacodemon", "baron-of-hell", "cyberdemon",
    "lost-soul", "pinky", "shotgun-guy", "specter",
    "spiderdemon", "zombieman",
}

# Umbral horizontal: si el enemigo esta dentro de este % del ancho, dispara.
ZONA_DISPARO = 0.20

# Contador interno para alternar exploracion (gira + avanza) sin girar en circulos.
_explorar_contador = 0
_GIROS_ANTES_AVANZAR = 6


def decidir(result, vida: int, ammo: int, frame_w: int) -> Action:
    global _explorar_contador

    centro_x = frame_w / 2
    enemigos = []

    if result is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            area = (x2 - x1) * (y2 - y1)
            if cls_name in ENEMIGOS:
                enemigos.append((cls_name, cx, area))

    # -- Sin enemigos visibles: explorar (avanzar + giros alternados) ----------
    if not enemigos:
        _explorar_contador += 1
        if _explorar_contador % (_GIROS_ANTES_AVANZAR + 1) == 0:
            return Action.MOVE_FORWARD
        return Action.TURN_RIGHT

    _explorar_contador = 0  # reset al ver un enemigo

    # -- Con vida muy baja: retroceder buscando distancia ----------------------
    if vida < 20:
        return Action.MOVE_BACKWARD

    # -- Sin ammo: avanzar (no puede disparar) ---------------------------------
    if ammo <= 0:
        return Action.MOVE_FORWARD

    # -- Elegir el enemigo mas cercano al centro (objetivo prioritario) --------
    objetivo = min(enemigos, key=lambda e: abs(e[1] - centro_x))
    _, obj_cx, obj_area = objetivo
    distancia_x = abs(obj_cx - centro_x)
    enemigo_grande = obj_area > (frame_w * frame_w * 0.05)  # >5% del frame = cerca

    # Zona de disparo: enemigo centrado
    if distancia_x < frame_w * ZONA_DISPARO:
        # Si esta cerca Y centrado, avanza mientras dispara para no dar tregua.
        if enemigo_grande:
            return Action.ATTACK
        # Si esta lejos pero centrado, avanza para acercarse
        return Action.MOVE_FORWARD if random.random() < 0.3 else Action.ATTACK

    # Fuera de la zona de disparo: girar hacia el objetivo
    return _girar_hacia(obj_cx, centro_x)


def _girar_hacia(target_x: float, centro_x: float) -> Action:
    return Action.TURN_LEFT if target_x < centro_x else Action.TURN_RIGHT
