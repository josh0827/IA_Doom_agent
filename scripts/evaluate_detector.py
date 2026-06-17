import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_YAML = ROOT / "dataset" / "doom-vizdoom" / "data.yaml"

from src.utils.paths import detector_weights


def main():
    try:
        WEIGHTS = detector_weights()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Evaluando: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA_YAML))
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
