import numpy as np

from src.policy.actions import Action
from src.policy.rules import decidir


class FakeBox:
    """Imita un box de YOLO: xyxy[0] es un array con metodo .tolist()."""

    def __init__(self, cls_idx, x1, x2):
        self.cls = np.array([cls_idx])
        self.xyxy = np.array([[x1, 0, x2, 100]])


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_sin_detecciones_avanza():
    result = FakeResult({0: "imp"}, [])
    assert decidir(result, vida=100, ammo=50, frame_w=640) == Action.MOVE_FORWARD


def test_enemigo_centrado_dispara():
    result = FakeResult({0: "imp"}, [FakeBox(0, 310, 330)])
    assert decidir(result, vida=100, ammo=50, frame_w=640) == Action.ATTACK


def test_enemigo_descentrado_gira():
    result = FakeResult({0: "imp"}, [FakeBox(0, 10, 40)])
    accion = decidir(result, vida=100, ammo=50, frame_w=640)
    assert accion == Action.TURN_LEFT
