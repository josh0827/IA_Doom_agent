from pathlib import Path
from ultralytics import YOLO


class Detector:
    def __init__(self, weights_path: Path, conf: float = 0.12):
        self.model = YOLO(str(weights_path))
        self.conf = conf

    def predict(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        return results[0]
