"""Diagnostico headless: corre el agente unos pasos e imprime que detecta y que decide."""
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.doom_env import DoomEnv
from src.perception.detector import Detector
from src.policy.actions import Action, action_to_vizdoom
from src.policy.rules import decidir, ENEMIGOS
from src.perception.visualization import draw_detections

WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"
OUT = ROOT / "diagnostico"


def main(n_steps: int = 40):
    OUT.mkdir(exist_ok=True)
    env = DoomEnv(SCENARIO, window_visible=False)
    detector = Detector(WEIGHTS, conf=0.15)

    # Botones disponibles segun el escenario realmente cargado
    print("=== BOTONES DISPONIBLES EN EL ESCENARIO ===")
    for b in env.game.get_available_buttons():
        print("  ", b)
    print()

    frame = env.reset()
    print(f"shape frame inicial: {None if frame is None else frame.shape}")
    info = {"vida": 100, "ammo": 50}

    for step in range(n_steps):
        if frame is None:
            print(f"[{step}] frame None -> fin")
            break
        result = detector.predict(frame)
        n = 0 if result is None else len(result.boxes)
        clases = []
        if n:
            for box in result.boxes:
                clases.append(f"{result.names[int(box.cls[0])]}:{float(box.conf[0]):.2f}")
        action = decidir(result, info["vida"], info["ammo"], frame.shape[1])
        print(f"[{step}] dets={n} {clases} -> accion={action.name} | vida={info['vida']} ammo={info['ammo']}")

        # Guarda los primeros frames anotados
        if step < 8:
            overlay = draw_detections(frame, result)
            cv2.imwrite(str(OUT / f"step_{step:02d}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        frame, reward, done, info = env.step(action_to_vizdoom(action))
        if done:
            print(f"[{step}] episodio terminado (reward final aplicado)")
            break

    env.close()
    print(f"\nFrames anotados guardados en: {OUT}")


if __name__ == "__main__":
    main()
