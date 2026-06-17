"""Demo del agente DQN entrenado jugando Doom (politica greedy, ventana visible).

Uso:
    python scripts/run_rl_agent.py                # juega en bucle hasta pulsar 'q'
    python scripts/run_rl_agent.py --episodes 10  # juega 10 partidas
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.frame_stack import FrameStack
from src.env.rl_env import RLEnv
from src.policy.rl_agent import DQNAgent
from src.policy.actions import Action
from src.perception.visualization import draw_detections
from src.utils.paths import detector_weights

WEIGHTS = detector_weights()
SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"
CKPT = ROOT / "runs" / "rl" / "dqn.pt"


def main(episodios: int = 0, max_steps: int = 400):
    """episodios=0 -> bucle infinito hasta pulsar 'q'."""
    if not CKPT.exists():
        print(f"Falta el agente entrenado: {CKPT}. Entrena con scripts/train_rl.py")
        return

    env = FrameStack(RLEnv(WEIGHTS, SCENARIO, frame_skip=2, window_visible=True), n_frames=3)
    agent = DQNAgent(env.state_dim, env.n_actions)
    agent.load(CKPT)
    print("Pulsa 'q' en la ventana del juego para salir.")

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
                        overlay, f"Ep {ep}  Accion: {Action(action).name}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    )
                    cv2.imshow("agente doom RL", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                if done:
                    break
            print(f"Episodio {ep}: reward={total:.1f} kills={info.get('kills', 0)}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=0,
                        help="numero de partidas (0 = bucle infinito hasta 'q')")
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()
    main(episodios=args.episodes, max_steps=args.max_steps)
