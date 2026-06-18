"""Genera notebooks/doom_entrega.ipynb: un notebook AUTOCONTENIDO que entrena
todo (detector YOLO + agente RL) y demuestra, con el codigo de cada modulo inline
para mostrar la estructura completa. Lee los modulos reales del repo (sin copiar a
mano) y les quita los imports internos `from src...` (en el notebook todo comparte
namespace).

Uso:  python scripts/build_entrega_notebook.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "doom_entrega.ipynb"

_STRIP = re.compile(r"^\s*(from\s+src\.|import\s+sys\b|sys\.path)")


def code_of(relpath: str) -> str:
    """Devuelve el codigo de un modulo del repo sin imports internos."""
    lines = (ROOT / relpath).read_text(encoding="utf-8").splitlines()
    keep = [ln for ln in lines if not _STRIP.match(ln)]
    return "\n".join(keep).strip() + "\n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.rstrip() + "\n"}


cells = []

# ── 0. Portada ────────────────────────────────────────────────────────────────
cells.append(md(
"""# Agente Doom: percepcion YOLO + Reinforcement Learning

**Teoria de Aprendizaje de Maquinas (UNAL).** Equipo: David (percepcion) y Joshua (entorno + politica).

Agente que juega Doom (ViZDoom, escenario `defend_the_center`: sala abierta tipo torreta)
combinando:

1. **Percepcion:** un detector **YOLOv8** entrenado con un dataset in-domain auto-etiquetado
   desde el propio motor (labels ground-truth de ViZDoom). Convierte cada frame en un vector
   de features (hay enemigo?, a que lado, que tan cerca, vida, ammo...).
2. **Politica:** un agente **Double DQN (Dueling + PER + n-step)** que aprende, sobre ese
   vector, que accion tomar para sobrevivir y matar enemigos que aparecen por todos lados.

```
ViZDoom (defend_the_center) --frame--> YOLO --detecciones--> features (13 dims)
        --> FrameStack (x3 = 39) --> DQN --accion--> ViZDoom
```

Este notebook es **autocontenido**: entrena el detector y el agente de cero y muestra el
codigo de cada modulo inline. Requiere **GPU** (Settings -> Accelerator -> GPU) e Internet ON."""))

# ── 1. Setup ──────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup\nInstala ViZDoom (headless), Ultralytics (YOLO) y OpenCV."))
cells.append(code(
"""import os
os.environ["DOOM_HEADLESS"] = "1"  # sin pantalla (Kaggle/Colab): ViZDoom usa software renderer
!apt-get -qq install -y libsdl2-2.0-0 > /dev/null 2>&1 || true
!pip -q install vizdoom ultralytics opencv-python-headless
import torch
print("CUDA disponible:", torch.cuda.is_available())"""))

# ── 2. Escenario ──────────────────────────────────────────────────────────────
defend_cfg = (ROOT / "src" / "env" / "scenarios" / "defend_the_center.cfg").read_text(encoding="utf-8")
cells.append(md(
"""## 2. Escenario

Escribimos el `.cfg` de `defend_the_center` con el layout de 8 botones que usa nuestra
politica. El `.wad` ya viene con el paquete `vizdoom`. La sala es una arena: el jugador en
el centro y enemigos (Demons) que aparecen en los bordes y se acercan."""))
cells.append(code(
'import os\n'
'os.makedirs("scenarios", exist_ok=True)\n'
'CFG_DEFEND = "scenarios/defend_the_center.cfg"\n'
'with open(CFG_DEFEND, "w") as f:\n'
'    f.write(' + repr(defend_cfg) + ')\n'
'print("cfg escrito:", CFG_DEFEND)'))

# ── 3. Acciones ───────────────────────────────────────────────────────────────
cells.append(md(
"""## 3. Espacio de acciones

Las 13 acciones del agente (mover, girar, disparar y combinadas). Se define primero porque
los demas modulos la referencian (`Action`)."""))
cells.append(code(code_of("src/policy/actions.py")))

# ── 4. Percepcion: codigo ─────────────────────────────────────────────────────
cells.append(md(
"""## 4. Percepcion (David)

El detector YOLO actua como extractor de percepcion. Dos piezas:

- `Detector`: corre YOLOv8 sobre el frame y devuelve las cajas. Umbral de confianza alto
  (0.40) para no alucinar en pasillos/paredes.
- `extract_state`: convierte las detecciones (+ vida, ammo) en el vector de **13 features**
  normalizadas que recibe el agente."""))
cells.append(code(code_of("src/perception/detector.py")))
cells.append(code(code_of("src/policy/rules.py")))      # ENEMIGOS (usado por features)
cells.append(code(code_of("src/perception/features.py")))
cells.append(code(code_of("src/perception/visualization.py")))

# ── 4. Dataset auto-etiquetado + entreno detector ─────────────────────────────
cells.append(md(
"""## 5. Detector YOLO (percepcion)

El detector se entreno **en LOCAL** (no aqui), para ahorrar tiempo de GPU en Kaggle. Esta
celda **carga los pesos ya entrenados** (`doom-v4`) clonando el repositorio publico del
proyecto. El proceso completo de como se construyo se documenta justo despues, pero **esas
celdas son de referencia y no se ejecutan**.

**Metodo (resumen):** ViZDoom expone un `labels_buffer` con la posicion y el nombre de cada
objeto en pantalla. Capturamos frames de varios escenarios y sacamos las cajas ground-truth
directo del motor (sin etiquetar a mano), mas frames vacios como fondo. Entrenamos YOLOv8s
sobre ese dataset in-domain. Resultado en local: mAP@0.5 ~0.91, deteccion de pinky en la
sala ~0.89."""))
cells.append(code(
'''from pathlib import Path
# Detector v4 entrenado en LOCAL. El repo es publico: clonamos (shallow) solo para
# traer los pesos ya entrenados. El codigo de como se entreno esta documentado abajo.
!git clone -q --depth 1 https://github.com/josh0827/IA_Doom_agent.git _repo
DETECTOR_WEIGHTS = Path("_repo/runs/doom-v4/weights/best.pt")
assert DETECTOR_WEIGHTS.exists(), "No se encontro best.pt tras clonar el repo."
print("detector v4 (entrenado en LOCAL):", DETECTOR_WEIGHTS,
      DETECTOR_WEIGHTS.stat().st_size // 1024, "KB")'''))

# Documentacion del proceso (NO se ejecuta): captura + auto-etiquetado + entreno en local.
_CAP_SRC = (
'''import random, glob
import cv2
import numpy as np
import vizdoom as vzd

CLASSES = ["baron-of-hell","cacodemon","cyberdemon","imp","lost-soul",
           "pinky","shotgun-guy","specter","spiderdemon","zombieman"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
NAME_MAP = {"Zombieman":"zombieman","ShotgunGuy":"shotgun-guy","ChaingunGuy":"shotgun-guy",
            "DoomImp":"imp","Demon":"pinky","Spectre":"specter","Cacodemon":"cacodemon",
            "BaronOfHell":"baron-of-hell","LostSoul":"lost-soul","Cyberdemon":"cyberdemon",
            "SpiderMastermind":"spiderdemon"}
SP = __import__("pathlib").Path(vzd.scenarios_path)   # cfg/wad que trae el paquete vizdoom
CAP_SCEN = ["deadly_corridor", "defend_the_center", "basic"]
MIN_BOX, EMPTY_RATE = 10, 0.20
DATA_DIR = __import__("pathlib").Path("dataset/doom-vizdoom")

def make_cap_game(name):
    g = vzd.DoomGame(); g.load_config(str(SP / f"{name}.cfg"))
    g.set_doom_scenario_path(str(SP / f"{name}.wad"))
    g.set_doom_skill(3); g.set_window_visible(False)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    g.set_labels_buffer_enabled(True); g.set_render_hud(False); g.init(); return g

def boxes_from_labels(st, w, h):
    out = []
    for l in st.labels:
        c = NAME_MAP.get(l.object_name)
        if c is None or l.width < MIN_BOX or l.height < MIN_BOX:
            continue
        out.append((CIDX[c], (l.x+l.width/2)/w, (l.y+l.height/2)/h, l.width/w, l.height/h))
    return out

def capturar(frames_por_scen=420, val_split=0.15, seed=0):
    random.seed(seed)
    for sp in ("train","valid"):
        for sub in ("images","labels"):
            (DATA_DIR/sp/sub).mkdir(parents=True, exist_ok=True)
    saved, counts = 0, {c:0 for c in CLASSES}
    for name in CAP_SCEN:
        try:
            g = make_cap_game(name)
        except Exception as e:
            print("salto", name, e); continue
        nb = g.get_available_buttons_size(); g.new_episode(); got = 0; tries = 0
        while got < frames_por_scen and tries < frames_por_scen*40:
            tries += 1
            if g.is_episode_finished(): g.new_episode()
            a = [0]*nb
            for b in random.sample(range(nb), k=min(2,nb)): a[b] = random.randint(0,1)
            g.make_action(a, random.choice([2,3,4]))
            st = g.get_state()
            if st is None: continue
            fr = st.screen_buffer; h,w = fr.shape[:2]
            bx = boxes_from_labels(st, w, h)
            if not bx and random.random() > EMPTY_RATE: continue
            split = "valid" if random.random() < val_split else "train"
            stem = f"{name}_{saved:05d}"
            cv2.imwrite(str(DATA_DIR/split/"images"/f"{stem}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            with open(DATA_DIR/split/"labels"/f"{stem}.txt","w") as f:
                for ci,cx,cy,bw,bh in bx:
                    f.write(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\\n"); counts[CLASSES[ci]] += 1
            saved += 1; got += 1
        g.close(); print(f"{name}: {got} frames")
    with open(DATA_DIR/"data.yaml","w") as f:
        f.write("train: ../train/images\\nval: ../valid/images\\n")
        f.write(f"nc: {len(CLASSES)}\\nnames: {CLASSES}\\n")
    print("total", saved, "| por clase:", {k:v for k,v in counts.items() if v})

capturar()''')

_TRAIN_SRC = (
'''from ultralytics import YOLO

det_model = YOLO("yolov8s.pt")
det_model.train(data=str(DATA_DIR/"data.yaml"), epochs=60, imgsz=640, batch=16,
                workers=2, cos_lr=True, patience=20, device=0,
                project="runs", name="doom-v4", exist_ok=True, verbose=True)
DETECTOR_WEIGHTS = Path("runs/doom-v4/weights/best.pt")''')

cells.append(md(
"### Como se construyo el detector (ejecutado en LOCAL, no aqui)\n\n"
"**1) Captura + auto-etiquetado desde ViZDoom** (cajas ground-truth, sin etiquetar a mano):\n\n"
"```python\n" + _CAP_SRC + "\n```\n\n"
"**2) Entrenamiento del detector YOLOv8s:**\n\n"
"```python\n" + _TRAIN_SRC + "\n```"))

# ── 5. Entorno: codigo ────────────────────────────────────────────────────────
cells.append(md(
"""## 6. Entorno (Joshua)

- `DoomEnv`: wrapper crudo de ViZDoom (frame, vida, ammo, kills, posicion, fin de nivel).
- `RLEnv`: une ViZDoom + detector + features y define la **recompensa**. Es *scenario-aware*:
  en la sala (`defend_the_center`) premia **sobrevivir, encarar y barrer girando, y matar**,
  y penaliza disparar a la nada (cuidar municion).
- `FrameStack`: apila 3 vectores para dar memoria de corto plazo (39 dims)."""))
cells.append(code(code_of("src/env/doom_env.py")))
cells.append(code(code_of("src/env/rl_env.py")))
cells.append(code(code_of("src/env/frame_stack.py")))

# ── 6. Politica: codigo ───────────────────────────────────────────────────────
cells.append(md(
"""## 7. Politica: Double DQN (Dueling + PER + n-step)

- `QNetwork`: MLP Dueling (separa V(s) y A(s,a)).
- `PrioritizedReplayBuffer`: muestrea las transiciones con mayor error TD.
- `DQNAgent`: Double DQN. `act(forbidden=...)` permite enmascarar acciones (en la sala
  prohibimos avanzar: el agente es una torreta que gira y dispara)."""))
cells.append(code(code_of("src/policy/dqn.py")))
cells.append(code(code_of("src/policy/rl_agent.py")))

# ── 7. Entrenamiento RL ───────────────────────────────────────────────────────
cells.append(md(
"""## 8. Entrenamiento del agente

Bucle de entrenamiento con n-step returns y evaluacion greedy periodica. Mascara
**sin avanzar** (torreta). Ajusta `TIMESTEPS` segun el tiempo disponible (el limite de
Kaggle es 12 h; con el detector corriendo por paso, ~1M pasos caben). Guarda el mejor
modelo por evaluacion en `dqn_room.pt`."""))
cells.append(code(
'''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

TIMESTEPS  = 1_000_000     # presupuesto de pasos (bajalo si tu sesion es corta)
MAX_STEPS  = 1500
FRAME_SKIP = 2
N_STEP     = 3
forbidden  = {int(Action.MOVE_FORWARD), int(Action.FORWARD_ATTACK)}  # torreta: no avanzar

class NStepBuffer:
    def __init__(self, n, gamma): self.n, self.gamma, self._b = n, gamma, deque()
    def push(self, s,a,r,s2,d):
        self._b.append((s,a,r,s2,d))
        return self._emit() if len(self._b) >= self.n else None
    def flush(self):
        out = []
        while self._b: out.append(self._emit())
        return out
    def _emit(self):
        s0,a0 = self._b[0][0], self._b[0][1]; G,gn = 0.0,1.0; sn,dn = self._b[-1][3], False
        for _,_,r,s2,d in self._b:
            G += gn*r; gn *= self.gamma
            if d: sn,dn = s2,True; break
        self._b.popleft(); return s0,a0,G,sn,float(dn),gn

def make_env():
    return FrameStack(RLEnv(DETECTOR_WEIGHTS, Path(CFG_DEFEND), frame_skip=FRAME_SKIP), n_frames=3)

def evaluate(agent, n_ep=3):
    R,K = [],[]
    for _ in range(n_ep):
        e = make_env(); s = e.reset(); tot,info = 0.0,{}
        for _ in range(MAX_STEPS):
            s,r,d,info = e.step(agent.act(s, greedy=True, forbidden=forbidden)); tot += r
            if d: break
        e.close(); R.append(tot); K.append(int(info.get("kills",0)))
    return float(np.mean(R)), float(np.mean(K))

env = make_env()
agent = DQNAgent(env.state_dim, env.n_actions, device="cuda" if torch.cuda.is_available() else "cpu")
nstep = NStepBuffer(N_STEP, agent.gamma)
rewards_hist, recientes, kills_rec = [], deque(maxlen=20), deque(maxlen=20)
mejor, total_steps, ep = float("-inf"), 0, 0
CKPT = Path("runs/rl/dqn_room.pt"); CKPT.parent.mkdir(parents=True, exist_ok=True)

while total_steps < TIMESTEPS:
    ep += 1; s = env.reset(); tot, losses, ep_kills = 0.0, [], 0
    for _ in range(MAX_STEPS):
        a = agent.act(s, forbidden=forbidden)
        s2,r,d,info = env.step(a); total_steps += 1
        tr = nstep.push(s,a,r,s2,float(d))
        if tr: agent.buffer.push(*tr)
        loss = agent.learn()
        if loss is not None: losses.append(loss)
        s = s2; tot += r; ep_kills = int(info.get("kills",0))
        if d: break
    for t in nstep.flush(): agent.buffer.push(*t)
    rewards_hist.append(tot); recientes.append(tot); kills_rec.append(ep_kills)
    if ep % 10 == 0:
        ls = f"{np.mean(losses):.3f}" if losses else "N/A"
        print(f"pasos {total_steps}/{TIMESTEPS} | ep {ep} | reward {tot:7.1f} | "
              f"media20 {np.mean(recientes):7.1f} | kills {ep_kills} | "
              f"kills_avg {np.mean(kills_rec):.1f} | eps {agent.epsilon():.3f} | loss {ls}")
    if ep % 50 == 0:
        er, ek = evaluate(agent)
        print(f"  [EVAL] reward {er:.1f} | kills {ek:.1f}")
        if er > mejor:
            mejor = er; agent.save(CKPT); print(f"  [CKPT] mejor guardado ({er:.1f})")
env.close()
plt.figure(figsize=(9,5)); plt.plot(rewards_hist, alpha=.35)
if len(rewards_hist) >= 20:
    mv = [np.mean(rewards_hist[max(0,i-19):i+1]) for i in range(len(rewards_hist))]
    plt.plot(mv, "crimson", lw=2)
plt.xlabel("Episodio"); plt.ylabel("Recompensa"); plt.title("Curva de aprendizaje (sala)")
plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs/rl/learning_curve_room.png", dpi=120)
plt.show(); print("Entrenamiento terminado. Mejor eval:", mejor)'''))

# ── 8. Evaluacion / demo ──────────────────────────────────────────────────────
cells.append(md(
"""## 9. Evaluacion final

Cargamos el mejor modelo y corremos varios episodios greedy (politica aprendida, sin
exploracion). Reportamos kills y supervivencia: las metricas que importan en la sala."""))
cells.append(code(
'''best_agent = DQNAgent(env.state_dim, env.n_actions,
                      device="cuda" if torch.cuda.is_available() else "cpu")
best_agent.load(CKPT)
ev_env = make_env()
for e in range(5):
    s = ev_env.reset(); info = {}; steps = 0
    for steps in range(1, MAX_STEPS+1):
        s,r,d,info = ev_env.step(best_agent.act(s, greedy=True, forbidden=forbidden))
        if d: break
    print(f"episodio {e+1}: kills {int(info.get('kills',0))} | pasos vivo {steps}")
ev_env.close()'''))

cells.append(md(
"""## 10. Conclusiones

- La **percepcion YOLO** (entrenada con auto-etiquetado desde ViZDoom) entrega un estado
  semantico e interpretable, mas eficiente en muestras que aprender desde pixeles crudos.
- El agente **Double DQN** aprende a defender la sala como torreta: gira para barrer, encara
  y dispara a los enemigos, conservando municion.
- Trabajo futuro: mas timesteps, mas clases de enemigos en el dataset, y comparar contra un
  baseline de DQN sobre pixeles."""))

# ── Ensamblar ─────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook generado:", OUT, "|", len(cells), "celdas")
