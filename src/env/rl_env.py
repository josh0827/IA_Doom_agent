"""Entorno RL: envuelve ViZDoom + detector YOLO.

Expone una interfaz tipo Gym (reset/step) donde la observacion es el vector de
features de YOLO (ver perception.features) y la recompensa combina la del juego
con shaping por bajas y por vida perdida. Aplica frame-skip para acelerar el
entrenamiento (YOLO infiere una vez por decision, no por tic).
"""
from pathlib import Path

import numpy as np
import vizdoom as vzd

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.perception.features import STATE_DIM, extract_state
from src.policy.actions import Action, action_to_vizdoom

# Balance de recompensa: matar debe ser mas rentable que solo correr.
PROGRESS_SCALE = 0.0   # sin recompensa por avanzar: matar es la UNICA forma de ganar
KILL_REWARD = 100.0    # premio fuerte por cada baja
HEALTH_PENALTY = 0.5   # penaliza recibir dano (no matar => recibir disparos)


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
        self._last_overlay_data = None  # (frame, result) para visualizacion
        self._prev_health = 100.0
        self._prev_kills = 0.0

    def _game_var(self, var) -> float:
        return float(self.env.game.get_game_variable(var))

    def reset(self) -> np.ndarray:
        frame = self.env.reset()
        self._prev_health = self._game_var(vzd.GameVariable.HEALTH)
        self._prev_kills = self._game_var(vzd.GameVariable.KILLCOUNT)
        ammo = self._game_var(vzd.GameVariable.AMMO2)
        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)
        return extract_state(result, self._prev_health, ammo, frame.shape[1])

    def step(self, action_idx: int):
        action = Action(action_idx)
        frame, reward_env, done, info = self.env.step(
            action_to_vizdoom(action), tics=self.frame_skip
        )
        reward = PROGRESS_SCALE * float(reward_env)

        if done or frame is None:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            self._last_overlay_data = None
            return next_state, reward, True, info

        # Shaping: premiar bajas, penalizar perdida de vida.
        kills, vida = info["kills"], info["vida"]
        reward += KILL_REWARD * max(0.0, kills - self._prev_kills)
        reward += HEALTH_PENALTY * (vida - self._prev_health)  # negativo si pierde vida
        self._prev_kills, self._prev_health = kills, vida

        result = self.detector.predict(frame)
        self._last_overlay_data = (frame, result)
        next_state = extract_state(result, vida, info["ammo"], frame.shape[1])
        return next_state, reward, False, info

    @property
    def last_overlay_data(self):
        """(frame, result) del ultimo paso, para dibujar detecciones en la demo."""
        return self._last_overlay_data

    def close(self):
        self.env.close()
