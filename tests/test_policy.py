from src.policy.actions import Action
from src.policy.rules import decidir


class FakeBox:
    def __init__(self, cls_idx, x1, x2):
        self.cls = [cls_idx]
        self.xyxy = [[x1, 0, x2, 100]]


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_sin_detecciones_gira():
    result = FakeResult({0: "imp"}, [])
    assert decidir(result, vida=100, ammo=50, frame_w=640) == Action.TURN_RIGHT


def test_enemigo_centrado_dispara():
    result = FakeResult({0: "imp"}, [FakeBox(0, 310, 330)])
    assert decidir(result, vida=100, ammo=50, frame_w=640) == Action.ATTACK


def test_sin_enemigos_avanza():
    result = FakeResult({0: "imp"}, [])
    assert decidir(result, vida=20, ammo=50, frame_w=640) == Action.TURN_RIGHT
