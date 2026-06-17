"""Agente Double DQN con Dueling architecture.

Mejoras respecto al DQN vanilla:
  - Double DQN: policy_net selecciona la accion, target_net evalua su Q-value.
    Esto corrige la sobreestimacion sistematica del DQN original.
  - Dueling QNetwork: separa V(s) y A(s,a), aprende mas rapido porque actualiza
    el valor del estado aunque no haya ejecutado todas las acciones.
  - Buffer 100k, lr 5e-4, eps_decay 50k pasos, n-step returns (gamma^N en el target).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.policy.dqn import QNetwork, PrioritizedReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 512,
        buffer_capacity: int = 100_000,
        target_update: int = 2000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=eps_decay_steps * 2, eta_min=lr * 0.1
        )
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(s).argmax(dim=1).item())

    def learn(self) -> float | None:
        """Double DQN con PER: loss ponderada por importance sampling."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, gammas, idxs, weights = \
            self.buffer.sample(self.batch_size)

        states      = torch.as_tensor(states,      device=self.device)
        actions     = torch.as_tensor(actions,     device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards,     device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones       = torch.as_tensor(dones,       device=self.device).unsqueeze(1)
        gammas      = torch.as_tensor(gammas,      device=self.device).unsqueeze(1)
        weights     = torch.as_tensor(weights,     device=self.device).unsqueeze(1)

        q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, best_actions)
            # gammas ya es gamma^N (N = pasos del retorno n-step de cada transicion),
            # el descuento correcto para arrancar el valor bootstrapeado.
            target = rewards + gammas * q_next * (1.0 - dones)

        td_errors = (q - target).abs().detach().cpu().numpy().squeeze()
        # Loss ponderada por IS weights para corregir el sesgo del PER
        loss = (weights * nn.functional.smooth_l1_loss(q, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(idxs, td_errors)

        self.steps += 1
        self.scheduler.step()
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), str(path))

    def load(self, path: Path):
        state = torch.load(str(path), map_location=self.device)
        self.policy_net.load_state_dict(state)
        self.target_net.load_state_dict(state)
