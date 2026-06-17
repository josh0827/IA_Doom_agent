"""Entrena el detector YOLO de enemigos Doom.

Mejoras sobre v1:
  - yolov8s (11M params) en lugar de yolov8n (3M) → +8-10 mAP estimados
  - 100 epocas con early stopping (patience=20)
  - cosine LR schedule para mejor convergencia al final
  - mixup + copy_paste ayudan a las clases con pocos ejemplos (pinky, specter, lost-soul)
  - cache=True acelera el entrenamiento si tienes suficiente RAM
  - Guarda en doom-v2 para no pisar los pesos anteriores

Uso:
    python scripts/train_detector.py
    python scripts/train_detector.py --nano    # forzar yolov8n si la RAM es justa
    python scripts/train_detector.py --epochs 150
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "dataset" / "doom-yolo" / "data.yaml"


def main(model_size: str = "s", epochs: int = 100):
    model = YOLO(f"yolov8{model_size}.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=640,
        batch=16,          # v1 usaba 8 — RTX 3050 Ti 4GB soporta 16 con yolov8s
        workers=0,         # 0 en Windows para evitar errores de multiprocessing
        cos_lr=True,       # LR coseno: baja suavemente al final
        patience=30,       # v1 tenia 20 — con 668 imagenes la curva es ruidosa
        cache="disk",      # preprocesa imagenes en disco: mas rapido sin consumir RAM
        amp=True,          # mixed precision float16: ~2x velocidad en RTX
        close_mosaic=15,   # ultimas 15 epocas sin mosaic para convergencia limpia
        # augmentacion extra para clases con pocos ejemplos
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        perspective=0.001,
        hsv_v=0.5,         # mas variacion de brillo — entornos oscuros de Doom
        device=0,          # GPU 0 (RTX) — cambiar a "cpu" si no hay GPU
        project=str(ROOT / "runs"),
        name="doom-v2",
        exist_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano",   action="store_true", help="usar yolov8n en lugar de yolov8s")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(model_size="n" if args.nano else "s", epochs=args.epochs)
