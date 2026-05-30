"""Demo visual en FreeDoom E1M1 con el agente DQN entrenado.

El agente recorre niveles reales de FreeDoom usando el modelo entrenado
en deadly_corridor. Pulsa 'q' en la ventana para salir.

Uso:
    python scripts/run_freedoom_demo.py
    python scripts/run_freedoom_demo.py --episodes 3
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.rl_env import RLEnv
from src.policy.rl_agent import DQNAgent
from src.policy.actions import Action
from src.perception.visualization import draw_detections

WEIGHTS  = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO = ROOT / "src" / "env" / "scenarios" / "freedoom.cfg"
CKPT     = ROOT / "runs" / "rl" / "dqn.pt"
WAD      = ROOT / "src" / "env" / "scenarios" / "freedoom1.wad"


def main(episodios: int = 0, max_steps: int = 2000):
    if not WAD.exists():
        print("Falta freedoom1.wad en src/env/scenarios/")
        print("Descargalo gratis desde: https://freedoom.github.io/")
        return
    if not CKPT.exists():
        print(f"Falta el agente entrenado: {CKPT}")
        print("Entrena con: python scripts/train_rl.py --episodes 400")
        return

    env = RLEnv(WEIGHTS, SCENARIO, frame_skip=4, window_visible=True)
    agent = DQNAgent(env.state_dim, env.n_actions)
    agent.load(CKPT)
    print("FreeDoom E1M1 | Pulsa 'q' para salir.")

    ep = 0
    try:
        while episodios == 0 or ep < episodios:
            ep += 1
            state = env.reset()
            total = 0.0
            info = {}
            for _ in range(max_steps):
                action = agent.act(state, greedy=True)
                state, reward, done, info = env.step(action)
                total += reward

                data = env.last_overlay_data
                if data is not None:
                    frame, result = data
                    overlay = draw_detections(frame, result)
                    cv2.putText(
                        overlay,
                        f"Ep {ep}  {Action(action).name}  kills={int(info.get('kills',0))}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
                    )
                    cv2.imshow("FreeDoom E1M1 - Agente RL", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                if done:
                    break
            print(f"Episodio {ep}: reward={total:.1f}  kills={info.get('kills', 0)}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    main(episodios=args.episodes, max_steps=args.max_steps)
