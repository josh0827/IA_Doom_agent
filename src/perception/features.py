"""Puente percepcion -> estado RL.

Convierte las detecciones de YOLO (+ vida y ammo) en un vector compacto de
features normalizadas que el agente DQN usa como observacion del entorno.
"""
import numpy as np

from src.policy.rules import ENEMIGOS

STATE_DIM = 8


def extract_state(result, health: float, ammo: float, frame_w: int) -> np.ndarray:
    """Devuelve un vector de 8 features normalizadas a partir de la deteccion YOLO.

    [enemy_present, nearest_offset, nearest_size, n_enemies,
     enemy_left, enemy_right, health_norm, ammo_norm]
    """
    centro_x = frame_w / 2
    enemigos = []  # (offset_x, area, cx)

    if result is not None and len(result.boxes) > 0:
        area_frame = float(frame_w * frame_w)  # normalizador estable
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            if cls_name not in ENEMIGOS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            enemigos.append((cx - centro_x, area / area_frame, cx))

    if enemigos:
        # enemigo mas centrado (objetivo natural)
        offset, _, cx = min(enemigos, key=lambda e: abs(e[0]))
        nearest_offset = float(np.clip(offset / centro_x, -1.0, 1.0))
        nearest_size = float(np.clip(max(e[1] for e in enemigos), 0.0, 1.0))
        n_enemies = float(np.clip(len(enemigos) / 5.0, 0.0, 1.0))
        enemy_left = 1.0 if cx < centro_x else 0.0
        enemy_right = 1.0 if cx >= centro_x else 0.0
        enemy_present = 1.0
    else:
        nearest_offset = nearest_size = n_enemies = 0.0
        enemy_left = enemy_right = enemy_present = 0.0

    health_norm = float(np.clip(health / 100.0, 0.0, 1.0))
    ammo_norm = float(np.clip(ammo / 50.0, 0.0, 1.0))

    return np.array(
        [enemy_present, nearest_offset, nearest_size, n_enemies,
         enemy_left, enemy_right, health_norm, ammo_norm],
        dtype=np.float32,
    )
