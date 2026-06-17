"""Genera un dataset YOLO in-domain a partir de ViZDoom, auto-etiquetando con el
labels_buffer del propio motor (cajas ground-truth, sin etiquetado manual).

Idea: el detector v2 se entreno con Doom clasico (Roboflow) y alucina en FreeDoom
(otro dominio). Aqui capturamos frames del juego real donde corre el agente y
sacamos las cajas exactas de cada enemigo desde ViZDoom. Incluimos frames vacios
(sin enemigos) como fondo para reducir falsos positivos en pasillos.

Salida: dataset/doom-vizdoom/{train,valid}/{images,labels} + data.yaml

Uso:
    python scripts/capture_dataset.py
    python scripts/capture_dataset.py --frames 1500 --val-split 0.15
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import vizdoom as vzd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCEN_DIR = ROOT / "src" / "env" / "scenarios"
OUT_DIR = ROOT / "dataset" / "doom-vizdoom"

# Mismo orden de clases que el dataset original (continuidad con ENEMIGOS / data.yaml).
CLASSES = ["baron-of-hell", "cacodemon", "cyberdemon", "imp", "lost-soul",
           "pinky", "shotgun-guy", "specter", "spiderdemon", "zombieman"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# Nombre del actor en ViZDoom -> clase del dataset.
NAME_MAP = {
    "Zombieman": "zombieman",
    "ShotgunGuy": "shotgun-guy",
    "ChaingunGuy": "shotgun-guy",   # variante humanoide con arma, visualmente cercana
    "DoomImp": "imp",
    "Demon": "pinky",
    "Spectre": "specter",
    "Cacodemon": "cacodemon",
    "BaronOfHell": "baron-of-hell",
    "LostSoul": "lost-soul",
    "Cyberdemon": "cyberdemon",
    "SpiderMastermind": "spiderdemon",
}

# Escenarios a recorrer. doom_skill alto -> mas enemigos en pantalla.
SCENARIOS = ["deadly_corridor", "freedoom", "basic", "defend_the_center"]

MIN_BOX_PX = 10          # descarta cajas mas pequenas que esto (enemigos tapados/lejisimos)
EMPTY_FRAME_RATE = 0.20  # fraccion de frames sin enemigos que conservamos como fondo


def find_wad(stem: str) -> Path | None:
    sp = Path(vzd.scenarios_path)
    for cand in (sp / f"{stem}.wad", SCEN_DIR / f"{stem}.wad", SCEN_DIR / "freedoom1.wad"):
        if cand.exists():
            return cand
    return None


def make_game(cfg_stem: str, skill: int = 3) -> vzd.DoomGame | None:
    cfg = SCEN_DIR / f"{cfg_stem}.cfg"
    if not cfg.exists():
        return None
    g = vzd.DoomGame()
    g.load_config(str(cfg))
    wad = find_wad(cfg_stem)
    if wad is not None:
        g.set_doom_scenario_path(str(wad))
    g.set_doom_skill(skill)
    g.set_window_visible(False)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    g.set_labels_buffer_enabled(True)
    g.set_render_hud(False)        # el HUD no aporta enemigos y mete ruido
    g.init()
    return g


def boxes_from_labels(state, w: int, h: int):
    """Devuelve lista de (cls_idx, cx, cy, bw, bh) normalizados [0,1]."""
    out = []
    for lab in state.labels:
        cls = NAME_MAP.get(lab.object_name)
        if cls is None:
            continue
        if lab.width < MIN_BOX_PX or lab.height < MIN_BOX_PX:
            continue
        cx = (lab.x + lab.width / 2) / w
        cy = (lab.y + lab.height / 2) / h
        bw = lab.width / w
        bh = lab.height / h
        # recorte de seguridad a [0,1]
        if not (0 < bw <= 1 and 0 < bh <= 1):
            continue
        out.append((CLASS_IDX[cls], np.clip(cx, 0, 1), np.clip(cy, 0, 1), bw, bh))
    return out


def capture(total_frames: int, val_split: float, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    for split in ("train", "valid"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    games = [(s, make_game(s)) for s in SCENARIOS]
    games = [(s, g) for s, g in games if g is not None]
    print("Escenarios activos:", [s for s, _ in games])
    if not games:
        print("No se pudo abrir ningun escenario."); return

    per_scen = total_frames // len(games)
    saved = 0
    empty_saved = 0
    cls_counts = {c: 0 for c in CLASSES}

    for scen, g in games:
        n_buttons = g.get_available_buttons_size()
        g.new_episode()
        grabbed = 0
        tries = 0
        max_tries = per_scen * 40  # tope para no quedar atrapado
        while grabbed < per_scen and tries < max_tries:
            tries += 1
            if g.is_episode_finished():
                g.new_episode()
            # accion: mayormente avanzar/girar para encarar enemigos, algo de azar
            a = [0] * n_buttons
            if n_buttons > 0:
                for b in random.sample(range(n_buttons), k=min(2, n_buttons)):
                    a[b] = random.randint(0, 1)
            g.make_action(a, random.choice([2, 3, 4]))
            st = g.get_state()
            if st is None:
                continue
            frame = st.screen_buffer  # (H, W, 3) RGB
            h, w = frame.shape[:2]
            boxes = boxes_from_labels(st, w, h)

            # Decide si guardar: con enemigos siempre; sin enemigos solo una fraccion
            if not boxes and random.random() > EMPTY_FRAME_RATE:
                continue
            if not boxes:
                empty_saved += 1

            split = "valid" if random.random() < val_split else "train"
            stem = f"{scen}_{saved:05d}"
            # ViZDoom da RGB; cv2.imwrite espera BGR
            cv2.imwrite(str(OUT_DIR / split / "images" / f"{stem}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            with open(OUT_DIR / split / "labels" / f"{stem}.txt", "w") as f:
                for cls_idx, cx, cy, bw, bh in boxes:
                    f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    cls_counts[CLASSES[cls_idx]] += 1
            saved += 1
            grabbed += 1
        g.close()
        print(f"  {scen}: {grabbed} frames")

    # data.yaml
    yaml = OUT_DIR / "data.yaml"
    with open(yaml, "w") as f:
        f.write("train: ../train/images\n")
        f.write("val: ../valid/images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print(f"\nTotal frames guardados: {saved}  (vacios/fondo: {empty_saved})")
    print("Instancias por clase:")
    for c in CLASSES:
        if cls_counts[c]:
            print(f"  {c:<14} {cls_counts[c]}")
    print(f"data.yaml -> {yaml}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=1400)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    capture(args.frames, args.val_split, args.seed)
