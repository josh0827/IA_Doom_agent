from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "doom-yolo" / "data.yaml"
WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"


def main():
    if not WEIGHTS.exists():
        print(f"No existe el modelo: {WEIGHTS}")
        print("Ejecuta primero scripts/train_detector.py")
        return
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(DATA_YAML))
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
