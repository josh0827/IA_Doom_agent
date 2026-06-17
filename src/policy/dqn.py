"""Componentes del DQN: red Q (Dueling MLP) y replay buffer.

Dueling architecture (Wang et al. 2016):
  Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]

Separar valor del estado (V) y ventaja por accion (A) acelera el aprendizaje
porque el agente puede actualizar V aunque no haya ejecutado todas las acciones.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Dueling MLP: tronco compartido -> cabezas Value y Advantage."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden // 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        v = self.value(features)
        a = self.advantage(features)
        # Q = V + A - mean(A)  =>  las Q siguen siendo comparables entre acciones
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Buffer uniforme — se conserva como fallback y para tests."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, gamma_n):
        self.buffer.append((state, action, reward, next_state, done, gamma_n))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, gammas = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(gammas, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── Prioritized Experience Replay ─────────────────────────────────────────────

class _SumTree:
    """Arbol binario de sumas para muestreo O(log N) proporcional a prioridades."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.n_entries = 0
        self._write = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(left + 1, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER (Schaul et al. 2016): muestrea transiciones con mayor TD-error mas frecuentemente.

    alpha  controla cuanto afecta la prioridad (0 = uniforme, 1 = puro PER).
    beta   corrige el sesgo de muestreo (importance sampling); sube de beta_start a 1.0.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 80_000,
    ):
        self.tree = _SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._frame = 0
        self._max_priority = 1.0

    def beta(self) -> float:
        return min(1.0, self.beta_start + self._frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done, gamma_n):
        self.tree.add(self._max_priority ** self.alpha,
                      (state, action, reward, next_state, done, gamma_n))

    def sample(self, batch_size: int):
        self._frame += 1
        segment = self.tree.total / batch_size
        batch, idxs, priorities = [], [], []
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                data = self.tree.data[0]
                idx = self.tree.capacity - 1
                p = max(self.tree.tree[idx], 1e-8)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        probs = np.array(priorities) / (self.tree.total + 1e-8)
        weights = (self.tree.n_entries * probs) ** (-self.beta())
        weights /= weights.max()
        states, actions, rewards, next_states, dones, gammas = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            np.array(gammas,      dtype=np.float32),
            idxs,
            np.array(weights,     dtype=np.float32),
        )

    def update_priorities(self, idxs: list, td_errors: np.ndarray):
        for idx, err in zip(idxs, td_errors):
            p = float(abs(err)) + 1e-6
            self._max_priority = max(self._max_priority, p)
            self.tree.update(idx, (p ** self.alpha))

    def __len__(self) -> int:
        return self.tree.n_entries
