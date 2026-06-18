"""Entorno RL: envuelve ViZDoom + detector YOLO.

Observacion: vector de 13 features (ver perception/features.py).

Recompensa (shaping):
  + KILL_REWARD     por cada baja confirmada (objetivo principal)
  + SHOOT_REWARD      senal densa por disparar cuando hay enemigo visible
  - WASTE_SHOT_PENALTY penaliza disparar sin enemigo visible (dispara a la nada)
  + STRAFE_REWARD   bonus por esquivar (strafe) cuando hay enemigo a la vista
  - HEALTH_PENALTY  penaliza recibir danio (delta de vida negativo)
  - DEATH_PENALTY   castigo explicito por morir
  - LIVING_COST     costo por paso: desincentiva merodear/farmear tiempo
  + PROGRESS_PER_UNIT  progreso POTENCIAL: premia solo terreno NUEVO ganado hacia
                    el chaleco (delta de la X maxima). NO es velocidad, asi que
                    merodear no farmea reward. Acotado por el largo del corredor.
  + GOAL_REWARD     premio terminal por COMPLETAR el nivel (alcanzar el chaleco
                    vivo, antes del timeout). Domina sobre cualquier farmeo.

Espacio de accion: 13 acciones (incluye strafe y acciones combinadas).
  0 MOVE_FORWARD        5 STRAFE_LEFT          10 TURN_LEFT_ATTACK
  1 MOVE_BACKWARD       6 STRAFE_RIGHT         11 TURN_RIGHT_ATTACK
  2 TURN_LEFT           7 FORWARD_ATTACK       12 BACKWARD_ATTACK
  3 TURN_RIGHT          8 STRAFE_LEFT_ATTACK
  4 ATTACK              9 STRAFE_RIGHT_ATTACK
"""
from pathlib import Path

import numpy as np

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.perception.features import STATE_DIM, extract_state
from src.policy.actions import Action, action_to_vizdoom
from src.policy.rules import ENEMIGOS

N_ACTIONS = 13

KILL_REWARD       = 150.0
SHOOT_REWARD      = 2.0    # disparar CON enemigo visible
WASTE_SHOT_PENALTY = 5.0   # disparar SIN enemigo visible (antes 1.5: rociaba paredes gratis)
STRAFE_REWARD     = 0.8    # esquivar con enemigo visible
HEALTH_PENALTY    = 0.8
DEATH_PENALTY     = 100.0  # morir duele (antes 50: rushear-y-morir salia rentable)
LIVING_COST       = 0.05   # costo por paso: desincentiva merodear/farmear tiempo
GOAL_REWARD       = 300.0  # completar el nivel: claramente mas que cualquier farmeo

# Progreso POTENCIAL: se premia solo el terreno NUEVO ganado hacia el chaleco
# (delta de la X maxima alcanzada), NO la velocidad de movimiento. Asi merodear o
# ir y venir NO acumula reward (no farmeable). Corredor mide ~1280 unidades en X,
# asi que un recorrido completo aporta ~1280 * PROGRESS_PER_UNIT.
PROGRESS_PER_UNIT = 0.2    # 1280 ud -> ~256 de progreso total (comparable a ~1.7 kills)

# ── Reward de SALA ABIERTA (defend_the_center, torreta) ────────────────────────
# Objetivo distinto: no hay chaleco/avance; hay que sobrevivir girando y matando
# enemigos que llegan de todos lados, cuidando municion.
SURVIVAL_BONUS = 0.1   # por paso vivo (en sala SI premiamos durar; sustituye LIVING_COST)
AIM_REWARD     = 0.2   # enemigo centrado (encararlo) -> fomenta apuntar girando
SCAN_REWARD    = 0.05  # girar cuando NO hay enemigo a la vista -> barrido izq-der
GOAL_ROOM      = 50.0  # sobrevivir hasta el timeout (aguantar toda la ronda)

# Confianza minima para que una deteccion cuente como "enemigo de combate" en la
# recompensa. Mas estricta que el umbral del detector (0.40): evita que un falso
# positivo marginal haga que disparar a la nada cobre SHOOT_REWARD en lugar de
# pagar WASTE_SHOT_PENALTY. Es el dial que rompe el "disparar a fantasmas".
COMBAT_CONF = 0.5

_STRAFE_ACTIONS = {Action.STRAFE_LEFT, Action.STRAFE_RIGHT,
                   Action.STRAFE_LEFT_ATTACK, Action.STRAFE_RIGHT_ATTACK,
                   Action.BACKWARD_ATTACK}  # kiting tambien es evasion
_SHOOT_ACTIONS  = {Action.ATTACK, Action.FORWARD_ATTACK,
                   Action.STRAFE_LEFT_ATTACK, Action.STRAFE_RIGHT_ATTACK,
                   Action.TURN_LEFT_ATTACK, Action.TURN_RIGHT_ATTACK,
                   Action.BACKWARD_ATTACK}
_TURN_ACTIONS   = {Action.TURN_LEFT, Action.TURN_RIGHT,
                   Action.TURN_LEFT_ATTACK, Action.TURN_RIGHT_ATTACK}

_MAX_DEPTH = 1000.0  # combate tipico ocurre a 200-800 unidades, mejor resolucion


class RLEnv:
    def __init__(
        self,
        weights: Path,
        scenario: Path,
        frame_skip: int = 2,
        conf: float = 0.40,  # ver nota en Detector: umbral alto contra falsos positivos
        window_visible: bool = False,
        skill: int | None = None,
    ):
        self.env = DoomEnv(scenario, window_visible=window_visible, skill=skill)
        self.detector = Detector(weights, conf=conf)
        # Sala abierta (defend_the_center) usa reward de torreta; pasillo usa progreso+meta.
        self._room = "defend" in Path(scenario).stem.lower()
        self.frame_skip = frame_skip
        self.state_dim = STATE_DIM
        self.n_actions = N_ACTIONS
        self._last_overlay_data = None
        self._prev_health = 100.0
        self._prev_kills  = 0.0
        self._steps_no_enemy = 0
        self._max_x = 0.0    # X maxima alcanzada (progreso potencial, no farmeable)
        self._frame_w = 640  # se actualiza en reset

    def reset(self) -> np.ndarray:
        frame = self.env.reset()
        self._prev_health    = self.env._last_info["vida"]
        self._prev_kills     = self.env._last_info["kills"]
        self._steps_no_enemy = 0
        self._max_x          = self.env._last_info.get("pos_x", 0.0)
        ammo = self.env._last_info["ammo"]

        if frame is not None:
            self._frame_w = frame.shape[1]

        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)
        return extract_state(result, self._prev_health, ammo,
                             self._frame_w, self._steps_no_enemy, 0.0,
                             self._enemy_distance(result))

    def step(self, action_idx: int):
        action = Action(action_idx)
        frame, reward_env, done, info = self.env.step(
            action_to_vizdoom(action), tics=self.frame_skip
        )
        # No usamos el reward nativo (premia velocidad => farmeable). Construimos
        # el shaping desde cero con progreso potencial + eventos.
        reward = 0.0

        if done or frame is None:
            if info.get("dead"):
                reward -= DEATH_PENALTY
            elif info.get("completed"):
                reward += GOAL_ROOM if self._room else GOAL_REWARD
            self._last_overlay_data = None
            state = np.zeros(self.state_dim, dtype=np.float32)
            return state, reward, True, info

        vida, ammo, kills = info["vida"], info["ammo"], info["kills"]
        delta_kills  = max(0.0, kills - self._prev_kills)
        delta_health = vida - self._prev_health           # negativo si recibio danio
        took_damage  = 1.0 if delta_health < 0 else 0.0

        # ── Percepcion: enemigo a la vista (conf alta) y centrado? ─────────────
        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)
        centro_x = self._frame_w / 2
        enemy_present = enemy_centered = False
        if result is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if (result.names[int(box.cls[0])] in ENEMIGOS
                        and float(box.conf[0]) >= COMBAT_CONF):
                    enemy_present = True
                    x1, _, x2, _ = box.xyxy[0].tolist()
                    if abs((x1 + x2) / 2 - centro_x) < 0.15 * self._frame_w:
                        enemy_centered = True

        # ── Eventos comunes (pasillo y sala) ───────────────────────────────────
        reward += KILL_REWARD * delta_kills
        reward += HEALTH_PENALTY * delta_health            # penaliza danio recibido
        if enemy_present and action in _SHOOT_ACTIONS:
            reward += SHOOT_REWARD                          # disparar a blanco real
        if not enemy_present and action in _SHOOT_ACTIONS:
            reward -= WASTE_SHOT_PENALTY                    # disparar a la nada
        if enemy_present and action in _STRAFE_ACTIONS:
            reward += STRAFE_REWARD                         # esquivar

        # ── Shaping especifico del modo ────────────────────────────────────────
        if self._room:
            reward += SURVIVAL_BONUS                        # sala: durar tiene valor
            if enemy_present and enemy_centered:
                reward += AIM_REWARD                        # encarar al enemigo
            if not enemy_present and action in _TURN_ACTIONS:
                reward += SCAN_REWARD                       # barrido buscando amenazas
        else:
            reward -= LIVING_COST                           # pasillo: no merodear
            pos_x = info.get("pos_x", 0.0)                  # progreso potencial (X nueva)
            if pos_x > self._max_x:
                reward += PROGRESS_PER_UNIT * (pos_x - self._max_x)
                self._max_x = pos_x

        self._steps_no_enemy = 0 if enemy_present else self._steps_no_enemy + 1
        self._prev_kills  = kills
        self._prev_health = vida

        next_state = extract_state(
            result, vida, ammo, self._frame_w,
            self._steps_no_enemy, took_damage,
            self._enemy_distance(result),
        )
        return next_state, reward, False, info

    def _enemy_distance(self, result) -> float:
        """Distancia normalizada al enemigo mas cercano usando el depth buffer."""
        depth = self.env.depth_buffer
        if depth is None or result is None or len(result.boxes) == 0:
            return 1.0  # desconocido = lejos
        centro_x = self._frame_w / 2
        best_cx, best_dist = None, float("inf")
        for box in result.boxes:
            if result.names[int(box.cls[0])] not in ENEMIGOS:
                continue
            x1, _, x2, _ = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            if abs(cx - centro_x) < best_dist:
                best_dist = abs(cx - centro_x)
                best_cx = int(np.clip(cx, 0, depth.shape[1] - 1))
        if best_cx is None:
            return 1.0
        cy = depth.shape[0] // 2
        dist = float(depth[cy, best_cx])
        return float(np.clip(dist / _MAX_DEPTH, 0.0, 1.0))

    @property
    def last_overlay_data(self):
        return self._last_overlay_data

    def close(self):
        self.env.close()
