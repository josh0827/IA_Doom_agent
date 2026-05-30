from pathlib import Path

from src.agent.runner import AgentRunner
from src.utils.metrics import FPSCounter

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"


def main(n_episodios: int = 5):
    agent = AgentRunner(WEIGHTS, SCENARIO, window_visible=False)
    fps = FPSCounter()
    recompensas = []
    try:
        for ep in range(n_episodios):
            total = 0.0
            for tick in agent.run_episode():
                total += tick["reward"]
                fps.tick()
            recompensas.append(total)
            print(f"Episodio {ep+1}: reward={total:.2f}")
    finally:
        agent.close()

    print(f"\nRecompensa media: {sum(recompensas)/len(recompensas):.2f}")
    print(f"FPS final: {fps.tick():.1f}")


if __name__ == "__main__":
    main()
