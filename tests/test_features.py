import numpy as np

from src.perception.features import STATE_DIM, extract_state


class FakeBox:
    """Imita un box de YOLO: cls[0] y xyxy[0].tolist()."""

    def __init__(self, cls_idx, x1, x2, y1=0, y2=100):
        self.cls = np.array([cls_idx])
        self.xyxy = np.array([[x1, y1, x2, y2]])


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def test_sin_detecciones_estado_vacio():
    state = extract_state(FakeResult({0: "imp"}, []), health=100, ammo=50, frame_w=640)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
    assert state[0] == 0.0  # enemy_present
    assert state[6] == 1.0  # health_norm
    assert state[7] == 1.0  # ammo_norm


def test_result_none_no_rompe():
    state = extract_state(None, health=50, ammo=25, frame_w=640)
    assert state[0] == 0.0
    assert np.isclose(state[6], 0.5)
    assert np.isclose(state[7], 0.5)


def test_enemigo_izquierda():
    result = FakeResult({0: "zombieman"}, [FakeBox(0, 10, 40)])
    state = extract_state(result, health=100, ammo=50, frame_w=640)
    assert state[0] == 1.0          # enemy_present
    assert state[1] < 0.0           # nearest_offset negativo = izquierda
    assert state[4] == 1.0          # enemy_left
    assert state[5] == 0.0          # enemy_right


def test_enemigo_centrado_offset_pequeno():
    result = FakeResult({0: "zombieman"}, [FakeBox(0, 310, 330)])
    state = extract_state(result, health=100, ammo=50, frame_w=640)
    assert state[0] == 1.0
    assert abs(state[1]) < 0.1      # casi centrado


def test_ignora_clases_no_enemigas():
    result = FakeResult({0: "barril"}, [FakeBox(0, 100, 150)])
    state = extract_state(result, health=100, ammo=50, frame_w=640)
    assert state[0] == 0.0          # no cuenta como enemigo


def test_valores_normalizados_en_rango():
    result = FakeResult({0: "imp"}, [FakeBox(0, 0, 640, 0, 480)])
    state = extract_state(result, health=200, ammo=999, frame_w=640)
    assert np.all(state >= -1.0) and np.all(state <= 1.0)
