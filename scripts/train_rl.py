"""Entrena el agente DQN sobre el entorno Doom (features YOLO).

Guarda los pesos en runs/rl/dqn.pt y la curva de aprendizaje en
runs/rl/learning_curve.png. Soporta reanudar desde un checkpoint.

Uso:
    python scripts/train_rl.py --episodes 400
    python scripts/train_rl.py --episodes 400 --resume --window  # ver mientras entrena
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.rl_env import RLEnv
from src.perception.visualization import draw_detections
from src.policy.actions import Action
from src.policy.rl_agent import DQNAgent

WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"
OUT_DIR = ROOT / "runs" / "rl"
CKPT = OUT_DIR / "dqn.pt"
CURVE = OUT_DIR / "learning_curve.png"


def plot_curve(rewards: list[float], path: Path, window: int = 20):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.35, label="reward por episodio")
    if len(rewards) >= window:
        media = [
            sum(rewards[max(0, i - window + 1): i + 1]) / len(rewards[max(0, i - window + 1): i + 1])
            for i in range(len(rewards))
        ]
        plt.plot(media, color="crimson", linewidth=2, label=f"media movil ({window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.title("Curva de aprendizaje DQN - Agente Doom")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=400, help="pasos (decisiones) por episodio")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reanudar desde checkpoint")
    parser.add_argument("--window", action="store_true", help="mostrar ventana mientras entrena")
    args = parser.parse_args()

    if not WEIGHTS.exists():
        print(f"Falta el modelo YOLO: {WEIGHTS}")
        return

    env = RLEnv(WEIGHTS, SCENARIO, frame_skip=args.frame_skip, window_visible=False)
    agent = DQNAgent(env.state_dim, env.n_actions)
    print(f"Dispositivo: {agent.device} | state_dim={env.state_dim} | n_actions={env.n_actions}")

    if args.resume and CKPT.exists():
        agent.load(CKPT)
        print(f"Reanudando desde {CKPT}")

    rewards_hist: list[float] = []
    recientes = deque(maxlen=20)
    mejor_media = float("-inf")

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            total = 0.0
            for _ in range(args.max_steps):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.buffer.push(state, action, reward, next_state, float(done))
                agent.learn()
                state = next_state
                total += reward

                if args.window:
                    data = env.last_overlay_data
                    if data is not None:
                        frame, result = data
                        overlay = draw_detections(frame, result)
                        cv2.putText(overlay,
                            f"Ep {ep} | eps={agent.epsilon():.2f} | {Action(action).name}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Entrenando DQN", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt

                if done:
                    break

            rewards_hist.append(total)
            recientes.append(total)
            media = sum(recientes) / len(recientes)
            print(
                f"Ep {ep:4d}/{args.episodes} | reward={total:8.1f} | "
                f"media20={media:8.1f} | eps={agent.epsilon():.3f} | steps={agent.steps}"
            )

            # Checkpoint del mejor modelo (por media movil) y curva periodica.
            if media > mejor_media and len(recientes) >= 10:
                mejor_media = media
                agent.save(CKPT)
            if ep % 20 == 0:
                plot_curve(rewards_hist, CURVE)
    finally:
        env.close()
        if args.window:
            cv2.destroyAllWindows()
        agent.save(OUT_DIR / "dqn_last.pt")
        plot_curve(rewards_hist, CURVE)
        print(f"\nGuardado: {CKPT} (mejor) y {OUT_DIR/'dqn_last.pt'} (ultimo)")
        print(f"Curva: {CURVE}")


if __name__ == "__main__":
    main()
