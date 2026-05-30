from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "doom-yolo" / "data.yaml"


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        project=str(ROOT / "runs"),
        name="doom-v1",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
