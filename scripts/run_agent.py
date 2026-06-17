from pathlib import Path

import cv2

from src.agent.runner import AgentRunner
from src.perception.visualization import draw_detections
from src.utils.paths import detector_weights

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = detector_weights()
SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"


def main():
    if not WEIGHTS.exists():
        print(f"Falta el modelo entrenado: {WEIGHTS}")
        return
    if not SCENARIO.exists():
        print(f"Falta el escenario: {SCENARIO}")
        return

    agent = AgentRunner(WEIGHTS, SCENARIO, window_visible=True)
    try:
        for tick in agent.run_episode(max_steps=2000):
            frame = tick["frame"]
            if frame is None:
                continue
            overlay = draw_detections(frame, tick["result"])
            cv2.imshow("agente doom", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        agent.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
