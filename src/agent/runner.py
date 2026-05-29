from pathlib import Path

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.policy.actions import action_to_vizdoom
from src.policy.rules import decidir


class AgentRunner:
    def __init__(self, weights: Path, scenario: Path, window_visible: bool = False):
        self.env = DoomEnv(scenario, window_visible=window_visible)
        self.detector = Detector(weights)

    def run_episode(self, max_steps: int = 2000):
        frame = self.env.reset()
        total_reward = 0.0
        info = {"vida": 100, "ammo": 50}

        for step in range(max_steps):
            if frame is None:
                break
            result = self.detector.predict(frame)
            action = decidir(result, info["vida"], info["ammo"], frame.shape[1])
            frame, reward, done, info = self.env.step(action_to_vizdoom(action))
            total_reward += reward
            yield {
                "step": step,
                "frame": frame,
                "result": result,
                "action": action,
                "reward": reward,
                "info": info,
            }
            if done:
                break

    def close(self):
        self.env.close()
