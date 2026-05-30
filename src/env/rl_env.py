"""Entorno RL: envuelve ViZDoom + detector YOLO.

Observacion: vector de 10 features (ver perception/features.py).
Recompensa:
  - KILL_REWARD     por cada baja confirmada
  - AIM_REWARD      por tener al enemigo apuntado (centrado en frame)
  - HEALTH_PENALTY  por recibir danio
  - living_reward   del escenario (penaliza tiempo muerto)
"""
from pathlib import Path

import numpy as np
import vizdoom as vzd

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.perception.features import STATE_DIM, extract_state
from src.policy.actions import Action, action_to_vizdoom

KILL_REWARD    = 100.0
AIM_REWARD     = 3.0   # bonus por cada paso con enemigo centrado (fomenta apuntar)
HEALTH_PENALTY = 0.5


class RLEnv:
    def __init__(
        self,
        weights: Path,
        scenario: Path,
        frame_skip: int = 4,
        conf: float = 0.08,
        window_visible: bool = False,
    ):
        self.env = DoomEnv(scenario, window_visible=window_visible)
        self.detector = Detector(weights, conf=conf)
        self.frame_skip = frame_skip
        self.state_dim = STATE_DIM
        self.n_actions = len(Action)
        self._last_overlay_data = None
        self._prev_health = 100.0
        self._prev_kills  = 0.0
        self._steps_no_enemy = 0
        self._frame_w = 640  # se actualiza en reset

    def reset(self) -> np.ndarray:
        frame = self.env.reset()
        self._prev_health     = self.env._last_info["vida"]
        self._prev_kills      = self.env._last_info["kills"]
        self._steps_no_enemy  = 0
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
        reward = 0.0   # PROGRESS_SCALE = 0: no reward por avanzar

        if done or frame is None:
            self._last_overlay_data = None
            state = np.zeros(self.state_dim, dtype=np.float32)
            return state, reward, True, info

        vida, ammo, kills = info["vida"], info["ammo"], info["kills"]

        # ── Reward shaping ────────────────────────────────────────────────────
        delta_kills  = max(0.0, kills - self._prev_kills)
        delta_health = vida - self._prev_health          # negativo si recibio danio
        took_damage  = 1.0 if delta_health < 0 else 0.0

        reward += KILL_REWARD    * delta_kills
        reward += HEALTH_PENALTY * delta_health          # penaliza danio recibido

        # Dense reward: bonus por apuntar al enemigo (enemigo centrado en frame)
        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)

        enemy_present = False
        enemy_centered = False
        centro_x = self._frame_w / 2

        if result is not None and len(result.boxes) > 0:
            from src.policy.rules import ENEMIGOS
            for box in result.boxes:
                cls_name = result.names[int(box.cls[0])]
                if cls_name in ENEMIGOS:
                    x1, _, x2, _ = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    enemy_present = True
                    if abs(cx - centro_x) < self._frame_w * 0.20:
                        enemy_centered = True
                    break

        if enemy_centered:
            reward += AIM_REWARD

        # Actualizar contadores
        if enemy_present:
            self._steps_no_enemy = 0
        else:
            self._steps_no_enemy += 1

        self._prev_kills  = kills
        self._prev_health = vida

        next_state = extract_state(
            result, vida, ammo, self._frame_w,
            self._steps_no_enemy, took_damage
        )
        return next_state, reward, False, info

    @property
    def last_overlay_data(self):
        return self._last_overlay_data

    def close(self):
        self.env.close()
