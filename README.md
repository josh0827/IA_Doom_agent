# Agente Doom con detector YOLO

Proyecto final de **Teoría de Aprendizaje de Máquinas** (UNAL, Facultad de Ingeniería).

Agente que juega Doom en tiempo real combinando un detector YOLO entrenado con dataset propio en Roboflow y una política basada en reglas sobre el entorno ViZDoom. La interfaz de despliegue es una web app en Streamlit.

## Equipo

- **David** — Percepción (dataset, YOLO, evaluación)
- **Joshua Marín Castrillón** — Entorno + política (ViZDoom, reglas de decisión)

## Arquitectura

```
Streamlit App
     │
Agent Runner (loop percibir → decidir → actuar)
     │
     ├── Perception (YOLO) ──┐
     │                       ├──→ Policy (reglas)
     └── ViZDoom env ←───────┘
```

## Stack

- Python 3.10+
- Ultralytics YOLOv8
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
src/perception/   detector YOLO y visualización
src/policy/       política de decisión (reglas)
src/env/          wrapper de ViZDoom
src/agent/        orquestación del loop principal
scripts/          entrenamiento, evaluación, captura
streamlit_app/    interfaz web
docs/             arquitectura y reporte académico
```

## Ejecución

Entrenar el detector:
```powershell
python scripts/train_detector.py
```

Correr el agente en modo debug (sin web):
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
