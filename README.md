# Agente Doom con detector YOLO

Proyecto final de **Teoría de Aprendizaje de Máquinas** (UNAL, Facultad de Ingeniería).

Agente que juega Doom en tiempo real combinando un detector YOLO entrenado con dataset propio en Roboflow y una política basada en reglas sobre el entorno ViZDoom. La interfaz de despliegue es una web app en Streamlit.

## Equipo

- **David** — Percepción (dataset, YOLO, evaluación)
- **Joshua Marín Castrillón** — Entorno + política (ViZDoom, reglas de decisión)

## Arquitectura

El detector YOLO actúa como **extractor de percepción**: convierte cada frame en un vector
de features (¿hay enemigo?, a qué lado, qué tan cerca, vida, ammo). Un agente **DQN
(Reinforcement Learning)** aprende, sobre ese vector, qué acción tomar para sobrevivir y
avanzar, mejorando episodio a episodio.

```
            ViZDoom env (deadly_corridor)
                   │ frame
                   ▼
        Perception (YOLO) ── detecciones
                   │
                   ▼
   features.py → vector de estado (8 dims)
                   │
                   ▼
        Policy DQN (aprende) ── acción
                   │
                   └────────────► ViZDoom (frame-skip)
```

Se conserva una **política de reglas** (`src/policy/rules.py`) como *baseline* para comparar
contra el agente que aprende.

## Stack

- Python 3.10+
- Ultralytics YOLOv8 (percepción)
- PyTorch (agente DQN)
- ViZDoom
- OpenCV
- Streamlit

## Setup local

```powershell
git clone https://github.com/josh0827/IA_Doom_agent.git
cd "proyecto final Agente doom"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Después descargar el dataset desde Roboflow (link pendiente) y colocar en `dataset/doom-yolo/`.

## Estructura

```
src/perception/   detector YOLO, visualización y extractor de features (features.py)
src/policy/       agente DQN (dqn.py, rl_agent.py) + política de reglas baseline (rules.py)
src/env/          wrapper de ViZDoom (doom_env.py) y entorno RL (rl_env.py)
src/agent/        orquestación del loop con reglas
scripts/          entrenamiento detector/RL, evaluación, demos
streamlit_app/    interfaz web
docs/             arquitectura y reporte académico
```

## Ejecución

Entrenar el detector:
```powershell
python scripts/train_detector.py
```

Entrenar el agente DQN (Reinforcement Learning):
```powershell
python scripts/train_rl.py --episodes 400
```
Genera `runs/rl/dqn.pt` (pesos) y `runs/rl/learning_curve.png` (curva de aprendizaje).

Ver el agente RL entrenado jugando:
```powershell
python scripts/run_rl_agent.py
```

Correr la baseline de reglas (sin aprendizaje):
```powershell
python scripts/run_agent.py
```

Lanzar la web app:
```powershell
streamlit run streamlit_app/app.py
```

## Documentación

- [Arquitectura](docs/arquitectura.md)
- [Decisiones técnicas](docs/decisiones-tecnicas.md)
- [Reporte final](docs/reporte-final.md)
