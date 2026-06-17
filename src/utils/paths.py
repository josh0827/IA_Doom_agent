"""Rutas centralizadas del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def detector_weights() -> Path:
    """Devuelve los pesos del detector mas reciente disponible (v3 > v2 > v1)."""
    for version in ("doom-v3", "doom-v2", "doom-v1"):
        w = ROOT / "runs" / version / "weights" / "best.pt"
        if w.exists():
            return w
    raise FileNotFoundError(
        "No se encontro ningun modelo YOLO entrenado.\n"
        "Ejecuta: python scripts/train_detector.py"
    )
