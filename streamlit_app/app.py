from pathlib import Path

import streamlit as st

from src.agent.runner import AgentRunner
from src.perception.visualization import draw_detections
from src.utils.metrics import FPSCounter

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO = ROOT / "src" / "env" / "scenarios" / "basic.cfg"

st.set_page_config(page_title="Agente Doom", layout="wide")
st.title("Agente Doom con YOLO + Politica")

col_video, col_stats = st.columns([3, 1])

with col_stats:
    max_steps = st.slider("Max steps", 100, 5000, 1500, 100)
    iniciar = st.button("Iniciar episodio")
    placeholder_vida = st.empty()
    placeholder_ammo = st.empty()
    placeholder_fps = st.empty()
    placeholder_reward = st.empty()

with col_video:
    placeholder_video = st.empty()


def main():
    if not WEIGHTS.exists():
        st.error(f"Falta el modelo en {WEIGHTS}. Entrena con scripts/train_detector.py")
        return
    agent = AgentRunner(WEIGHTS, SCENARIO, window_visible=False)
    fps = FPSCounter()
    total_reward = 0.0
    try:
        for tick in agent.run_episode(max_steps=max_steps):
            frame = tick["frame"]
            if frame is None:
                continue
            overlay = draw_detections(frame, tick["result"])
            placeholder_video.image(overlay, channels="RGB", use_column_width=True)
            total_reward += tick["reward"]
            placeholder_vida.metric("Vida", tick["info"]["vida"])
            placeholder_ammo.metric("Ammo", tick["info"]["ammo"])
            placeholder_fps.metric("FPS", f"{fps.tick():.1f}")
            placeholder_reward.metric("Reward total", f"{total_reward:.1f}")
    finally:
        agent.close()


if iniciar:
    main()
