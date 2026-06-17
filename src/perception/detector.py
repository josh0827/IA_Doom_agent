from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO


class Detector:
    # conf alto a proposito: el detector tiene precision ~0.43, asi que un umbral
    # bajo inunda la escena de falsos positivos. 0.40 corta el grueso del ruido.
    def __init__(self, weights_path: Path, conf: float = 0.40):
        self.model = YOLO(str(weights_path))
        self.conf = conf
        # Determinar si hay GPU disponible para mover frames directamente
        self._use_gpu = torch.cuda.is_available()
        if self._use_gpu:
            self.model.to("cuda")

    def predict(self, frame):
        if frame is None:
            return None
        if self._use_gpu:
            # Convertir (H, W, 3) numpy uint8 a tensor CUDA float16 normalizado
            # Evita la transferencia CPU->GPU frame a frame dentro de YOLO
            t = torch.from_numpy(np.ascontiguousarray(frame)).cuda()
            t = t.permute(2, 0, 1).unsqueeze(0).half() / 255.0
            results = self.model.predict(t, conf=self.conf, verbose=False, half=True)
        else:
            # Ultralytics asume que un numpy HWC viene en BGR (convencion OpenCV) y
            # lo convierte a RGB internamente. ViZDoom entrega RGB24, asi que lo
            # pasamos a BGR para que esa conversion interna lo deje en RGB real.
            # Sin esto, en CPU el modelo veria los canales R y B intercambiados.
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            results = self.model.predict(bgr, conf=self.conf, verbose=False)
        return results[0]
