"""Entorno RL: envuelve ViZDoom + detector YOLO.

Observacion: vector de 10 features (ver perception/features.py).

Recompensa (shaping):
  + KILL_REWARD     por cada baja confirmada (objetivo principal)
  + SHOOT_REWARD    senal densa por disparar cuando hay enemigo visible
  + HEALTH_PENALTY  penaliza recibir danio (delta de vida negativo)
  - DEATH_PENALTY   castigo explicito por morir
  + PROGRESS_SCALE  fraccion del reward nativo del escenario (avanzar por el
                    corredor); incluye living_reward y death_penalty nativos,
                    pero atenuados para que NO dominen sobre matar.

Espacio de accion: 5 acciones utiles (MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT,
TURN_RIGHT, ATTACK). Se excluyen USE e IDLE (inutiles en el corredor y la
segunda haria que el agente se congele recibiendo disparos).
"""
from pathlib import Path

import numpy as np

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.perception.features import STATE_DIM, extract_state
from src.policy.actions import Action, action_to_vizdoom
from src.policy.rules import ENEMIGOS

N_ACTIONS = 5          # Action 0..4 = FORWARD, BACKWARD, LEFT, RIGHT, ATTACK

KILL_REWARD    = 100.0
SHOOT_REWARD   = 1.0   # bonus suave por disparar con enemigo a la vista
HEALTH_PENALTY = 0.5   # por punto de vida perdido
DEATH_PENALTY  = 30.0  # castigo explicito al morir
PROGRESS_SCALE = 0.05  # incentivo suave de avanzar (reward nativo del escenario)


class RLEnv:
    def __init__(
        self,
        weights: Path,
        scenario: Path,
        frame_skip: int = 4,
        conf: float = 0.12,
        window_visible: bool = False,
    ):
        self.env = DoomEnv(scenario, window_visible=window_visible)
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
                             self._frame_w, self._steps_no_enemy, 0.0)

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

        # ── Percepcion: hay enemigo a la vista? ───────────────────────────────
        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)

        enemy_present = False
        if result is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if result.names[int(box.cls[0])] in ENEMIGOS:
                    enemy_present = True
                    break

        # Senal densa: disparar cuando hay enemigo visible guia hacia el kill.
        if enemy_present and action == Action.ATTACK:
            reward += SHOOT_REWARD

        self._steps_no_enemy = 0 if enemy_present else self._steps_no_enemy + 1
        self._prev_kills  = kills
        self._prev_health = vida

        next_state = extract_state(
            result, vida, ammo, self._frame_w,
            self._steps_no_enemy, took_damage,
        )
        return next_state, reward, False, info

    @property
    def last_overlay_data(self):
        return self._last_overlay_data

    def close(self):
        self.env.close()
