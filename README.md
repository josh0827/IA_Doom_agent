# Agente Doom: percepción YOLO + Reinforcement Learning

Proyecto final de **Teoría de Aprendizaje de Máquinas** — Universidad Nacional de Colombia.

Un agente que juega Doom (ViZDoom) combinando dos modelos de aprendizaje:

1. **Percepción:** un detector **YOLOv8** que convierte cada frame del juego en información
   semántica (qué enemigos hay, dónde y a qué distancia).
2. **Política:** un agente **Double DQN** (Dueling + Prioritized Experience Replay + n-step)
   que aprende, sobre esa información, qué acción tomar para sobrevivir y combatir.

La idea central del diseño es **separar percepción de control**: en lugar de aprender
directamente desde los píxeles, el agente decide sobre un vector compacto e interpretable,
lo que hace el aprendizaje mucho más eficiente en muestras.

## Equipo

- **Joshua Marín Castrillón** — Entorno + política (ViZDoom, recompensa, agente RL).
- **David Henao Rojas** — Percepción (dataset, entrenamiento y evaluación de YOLO).

## Arquitectura

```
ViZDoom ──frame──▶ YOLOv8 ──cajas──▶ features (13 dims) ──▶ FrameStack ×3 (39 dims)
                                                                    │
                                                                    ▼
ViZDoom ◀──acción── Double DQN (Dueling + PER + n-step) ◀──────────┘
```

- **Estado:** 13 features normalizadas por frame (presencia de enemigo, lado, cercanía,
  vida, munición, distancia por *depth buffer*...), apiladas en 3 frames = **39 dimensiones**.
- **Acciones:** 13 discretas (mover, girar, disparar y combinadas).
- **Recompensa:** diseñada por escenario (*reward shaping*), no la nativa de ViZDoom.

## Organización del repositorio

El código está separado por responsabilidad. Cada carpeta tiene un propósito único:

```
src/                    Código fuente (librería del proyecto)
├── perception/         PERCEPCIÓN (módulo de David)
│   ├── detector.py       carga YOLOv8 y corre la detección sobre el frame
│   ├── features.py       convierte detecciones → vector de estado (13 dims)
│   └── visualization.py  dibuja las cajas sobre el frame (overlay)
├── policy/             POLÍTICA / DECISIÓN
│   ├── actions.py        las 13 acciones del agente
│   ├── dqn.py            red Q (Dueling MLP) + Prioritized Replay Buffer
│   ├── rl_agent.py       agente Double DQN (act, learn, save/load)
│   └── rules.py          baseline de reglas + clases de enemigos fiables
├── env/                ENTORNO
│   ├── doom_env.py       wrapper crudo de ViZDoom (frame, vida, ammo, kills)
│   ├── rl_env.py         entorno RL: une ViZDoom + detector + recompensa
│   ├── frame_stack.py    apila 3 estados (memoria de corto plazo)
│   └── scenarios/        configuraciones .cfg de cada escenario
├── agent/              orquestación del loop con la baseline de reglas
└── utils/              rutas, métricas y utilidades

scripts/                Scripts ejecutables (entrenar, evaluar, demos)
├── capture_dataset.py    genera el dataset auto-etiquetado desde ViZDoom
├── train_detector.py     entrena el detector YOLO
├── evaluate_detector.py  métricas del detector (mAP, matriz de confusión)
├── train_rl.py           entrena el agente RL
└── run_rl_agent.py       corre el agente entrenado (ventana en vivo)

notebooks/              Entregables reproducibles
├── doom_entrega.ipynb    notebook autocontenido (todo el código inline)
└── kaggle_train_room.ipynb  entrenamiento RL de la sala en Kaggle (GPU)

streamlit_app/          Demo web interactiva (app.py)
sustentacion/           Diapositivas (PDF) y guía de estudio de la sustentación
tests/                  Pruebas unitarias (features y política)
runs/                   Modelos entrenados versionados (detector y agente)
```

### Por qué está así

- **`src/` separa percepción, política y entorno** en módulos independientes. Refleja el
  reparto del trabajo (David: `perception/`, Joshua: `env/` + `policy/`) y permite probar
  cada pieza por separado.
- **`scripts/` contiene los puntos de entrada** (entrenar, evaluar, demo). No tienen lógica
  propia: orquestan los módulos de `src/`.
- **`notebooks/` son los entregables** que corren de punta a punta en Kaggle.
- **`runs/` versiona los modelos ya entrenados** (detector `doom-v4` y políticas) para
  reproducir resultados sin reentrenar.

## Stack

Python 3.10+ · Ultralytics YOLOv8 · PyTorch · ViZDoom · OpenCV · Streamlit.

## Resultados

- **Detector (`doom-v4`):** mAP@0.5 = **0.908** en validación, sobre un dataset in-domain de
  ~2.200 imágenes auto-etiquetadas desde el motor (sin anotación manual).
- **Agente de sala:** entrenado 1.000.000 de pasos; la recompensa media móvil crece de ~300
  a ~1.400 a lo largo del entrenamiento.

## Cómo se entrenó el detector (sin etiquetar a mano)

ViZDoom expone un `labels_buffer` con la posición y el nombre exactos de cada objeto en
pantalla. `scripts/capture_dataset.py` recorre varios escenarios y extrae las cajas
ground-truth directamente del motor, generando un dataset YOLO sin anotación manual.

## Setup local

```powershell
git clone https://github.com/josh0827/IA_Doom_agent.git
cd "proyecto final Agente doom"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Generar el dataset auto-etiquetado y entrenar el detector:
```powershell
python scripts/capture_dataset.py
python scripts/train_detector.py
```

Entrenar el agente RL (pasillo o sala):
```powershell
python scripts/train_rl.py --scenario deadly_corridor                 # pasillo
python scripts/train_rl.py --scenario defend_the_center --no-forward   # sala (torreta)
```

Ver el agente entrenado jugando:
```powershell
python scripts/run_rl_agent.py --scenario defend_the_center --no-forward
```

Lanzar la demo web (3 escenarios: pasillo, sala real y sala demo limpia):
```powershell
streamlit run streamlit_app/app.py
```

## Entregables

- **Notebook:** [`notebooks/doom_entrega.ipynb`](notebooks/doom_entrega.ipynb) — autocontenido,
  muestra el código de cada módulo y entrena el agente en GPU.
- **Sustentación:** [`sustentacion/`](sustentacion/) — diapositivas en PDF con la matemática
  del Double DQN y la guía de estudio.
