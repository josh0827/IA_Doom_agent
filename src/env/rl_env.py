"""Entorno RL: envuelve ViZDoom + detector YOLO.

Observacion: vector de 13 features (ver perception/features.py).

Recompensa (shaping):
  + KILL_REWARD     por cada baja confirmada (objetivo principal)
  + SHOOT_REWARD      senal densa por disparar cuando hay enemigo visible
  - WASTE_SHOT_PENALTY penaliza disparar sin enemigo visible (dispara a la nada)
  + STRAFE_REWARD   bonus por esquivar (strafe) cuando hay enemigo a la vista
  - HEALTH_PENALTY  penaliza recibir danio (delta de vida negativo)
  - DEATH_PENALTY   castigo explicito por morir
  + SURVIVAL_BONUS  premio minimo por cada paso vivo (incentiva no suicidarse)
  + PROGRESS_SCALE  fraccion del reward nativo del escenario = avanzar hacia el
                    chaleco del final del corredor (senal densa de progreso);
                    incluye living_reward y death_penalty nativos.
  + GOAL_REWARD     premio terminal por COMPLETAR el nivel (alcanzar el chaleco
                    vivo, antes del timeout). Hace explicito "llegar al final".

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
SHOOT_REWARD      = 2.0   # disparar CON enemigo visible
WASTE_SHOT_PENALTY = 1.5  # disparar SIN enemigo visible (se resta)
STRAFE_REWARD     = 0.8   # esquivar con enemigo visible
HEALTH_PENALTY    = 0.8
DEATH_PENALTY     = 50.0
SURVIVAL_BONUS    = 0.02
PROGRESS_SCALE    = 0.10   # peso del avance hacia el chaleco (antes 0.02: casi invisible)
GOAL_REWARD       = 150.0  # completar el nivel (llegar al chaleco vivo) = tan valioso como un kill

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
        self.frame_skip = frame_skip
        self.state_dim = STATE_DIM
        self.n_actions = N_ACTIONS
        self._last_overlay_data = None
        self._prev_health = 100.0
        self._prev_kills  = 0.0
        self._steps_no_enemy = 0
        self._frame_w = 640  # se actualiza en reset

    def reset(self) -> np.ndarray:
        frame = self.env.reset()
        self._prev_health    = self.env._last_info["vida"]
        self._prev_kills     = self.env._last_info["kills"]
        self._steps_no_enemy = 0
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
        # Reward nativo del escenario (progreso + living + death), atenuado.
        reward = PROGRESS_SCALE * float(reward_env)

        if done or frame is None:
            if info.get("dead"):
                reward -= DEATH_PENALTY
            elif info.get("completed"):
                reward += GOAL_REWARD   # llego al chaleco del final: objetivo logrado
            self._last_overlay_data = None
            state = np.zeros(self.state_dim, dtype=np.float32)
            return state, reward, True, info

        vida, ammo, kills = info["vida"], info["ammo"], info["kills"]

        # ── Combate ───────────────────────────────────────────────────────────
        delta_kills  = max(0.0, kills - self._prev_kills)
        delta_health = vida - self._prev_health           # negativo si recibio danio
        took_damage  = 1.0 if delta_health < 0 else 0.0

        reward += KILL_REWARD * delta_kills
        reward += HEALTH_PENALTY * delta_health           # penaliza danio recibido
        reward += SURVIVAL_BONUS                          # sobrevivir tiene valor

        # ── Percepcion: hay enemigo a la vista? ───────────────────────────────
        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)

        enemy_present = False
        if result is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if (result.names[int(box.cls[0])] in ENEMIGOS
                        and float(box.conf[0]) >= COMBAT_CONF):
                    enemy_present = True
                    break

        # Disparar CON enemigo visible: senal densa hacia el kill.
        if enemy_present and action in _SHOOT_ACTIONS:
            reward += SHOOT_REWARD

        # Disparar SIN enemigo visible: penalizar (dispara a la nada).
        if not enemy_present and action in _SHOOT_ACTIONS:
            reward -= WASTE_SHOT_PENALTY

        # Strafe con enemigo visible: comportamiento de evasion humano.
        if enemy_present and action in _STRAFE_ACTIONS:
            reward += STRAFE_REWARD

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
