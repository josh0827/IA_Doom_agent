"""Puente percepcion -> estado RL.

Convierte las detecciones de YOLO (+ vida, ammo y senales de combate) en un
vector compacto de features normalizadas que el agente DQN usa como observacion.

STATE_DIM = 10:
  0  enemy_present        1 si hay >= 1 enemigo detectado
  1  nearest_offset       offset horizontal del enemigo mas centrado [-1, 1]
  2  nearest_size         area relativa del bbox mayor [0, 1]
  3  n_enemies            nº de enemigos / 5 (cap 1.0)
  4  enemy_left           1 si el enemigo mas cercano esta a la izquierda
  5  enemy_right          1 si esta a la derecha
  6  health_norm          vida / 100
  7  ammo_norm            ammo / 50 (cap 1.0)
  8  steps_no_enemy_norm  pasos sin ver enemigo / 50 (urgencia de explorar)
  9  took_damage          1 si recibio danio en el ultimo paso
"""
import numpy as np

from src.policy.rules import ENEMIGOS

STATE_DIM = 10


def extract_state(
    result,
    health: float,
    ammo: float,
    frame_w: int,
    steps_no_enemy: int = 0,
    took_damage: float = 0.0,
) -> np.ndarray:
    """Devuelve vector de 10 features normalizadas."""
    centro_x = frame_w / 2
    enemigos = []

    if result is not None and len(result.boxes) > 0:
        area_frame = float(frame_w * frame_w)
        for box in result.boxes:
            cls_name = result.names[int(box.cls[0])]
            if cls_name not in ENEMIGOS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            enemigos.append((cx - centro_x, area / area_frame, cx))

    if enemigos:
        offset, _, cx = min(enemigos, key=lambda e: abs(e[0]))
        nearest_offset     = float(np.clip(offset / centro_x, -1.0, 1.0))
        nearest_size       = float(np.clip(max(e[1] for e in enemigos), 0.0, 1.0))
        n_enemies          = float(np.clip(len(enemigos) / 5.0, 0.0, 1.0))
        enemy_left         = 1.0 if cx < centro_x else 0.0
        enemy_right        = 1.0 if cx >= centro_x else 0.0
        enemy_present      = 1.0
    else:
        nearest_offset = nearest_size = n_enemies = 0.0
        enemy_left = enemy_right = enemy_present = 0.0

    health_norm        = float(np.clip(health / 100.0, 0.0, 1.0))
    ammo_norm          = float(np.clip(ammo / 50.0, 0.0, 1.0))
    steps_no_enemy_norm = float(np.clip(steps_no_enemy / 50.0, 0.0, 1.0))
    took_damage_f      = float(np.clip(took_damage, 0.0, 1.0))

    return np.array(
        [enemy_present, nearest_offset, nearest_size, n_enemies,
         enemy_left, enemy_right, health_norm, ammo_norm,
         steps_no_enemy_norm, took_damage_f],
        dtype=np.float32,
    )
