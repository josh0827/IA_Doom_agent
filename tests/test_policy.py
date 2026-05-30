import numpy as np

from src.policy.actions import Action
from src.policy.rules import decidir


class FakeBox:
    """Imita un box de YOLO: xyxy[0] es un array con metodo .tolist()."""

    def __init__(self, cls_idx, x1, x2, y1=0, y2=100):
        self.cls = np.array([cls_idx])
        self.xyxy = np.array([[x1, y1, x2, y2]])


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_sin_detecciones_explora():
    result = FakeResult({0: "imp"}, [])
    # Sin enemigos: gira o avanza (exploracion)
    accion = decidir(result, vida=100, ammo=50, frame_w=640)
    assert accion in (Action.TURN_RIGHT, Action.MOVE_FORWARD)


def test_enemigo_centrado_dispara():
    result = FakeResult({0: "imp"}, [FakeBox(0, 290, 350, 0, 300)])
    # Enemigo grande y centrado: dispara
    accion = decidir(result, vida=100, ammo=50, frame_w=640)
    assert accion in (Action.ATTACK, Action.MOVE_FORWARD)


def test_enemigo_descentrado_gira():
    result = FakeResult({0: "imp"}, [FakeBox(0, 10, 40)])
    accion = decidir(result, vida=100, ammo=50, frame_w=640)
    assert accion == Action.TURN_LEFT


def test_vida_baja_retrocede():
    result = FakeResult({0: "zombieman"}, [FakeBox(0, 300, 340)])
    accion = decidir(result, vida=15, ammo=50, frame_w=640)
    assert accion == Action.MOVE_BACKWARD


def test_sin_ammo_avanza():
    result = FakeResult({0: "zombieman"}, [FakeBox(0, 300, 340)])
    accion = decidir(result, vida=100, ammo=0, frame_w=640)
    assert accion == Action.MOVE_FORWARD
