"""FrameStack: apila N vectores de estado consecutivos.

Sin esto el agente solo ve el momento actual y no puede saber si un enemigo
se acerca, si su vida esta cayendo rapido, o si acaba de girar. Apilar 3 frames
le da memoria de corto plazo sin la complejidad de una red recurrente (LSTM).

El input al DQN pasa de STATE_DIM a STATE_DIM * N_FRAMES (ej: 13*3 = 39).
Los frames mas nuevos van al final del vector.
"""
from collections import deque

import numpy as np


class FrameStack:
    def __init__(self, env, n_frames: int = 3):
        self.env = env
        self.n_frames = n_frames
        self._frames = deque(maxlen=n_frames)
        self.state_dim = env.state_dim * n_frames
        self.n_actions = env.n_actions

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        # Rellena con el primer estado para que no haya ceros al inicio
        for _ in range(self.n_frames):
            self._frames.append(state)
        return self._obs()

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        self._frames.append(state)
        return self._obs(), reward, done, info

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), dtype=np.float32)

    @property
    def last_overlay_data(self):
        return self.env.last_overlay_data

    def close(self):
        self.env.close()
