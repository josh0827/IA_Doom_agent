from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS   = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"
SCENARIO  = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"
CKPT_RL   = ROOT / "runs" / "rl" / "dqn.pt"
CURVE_IMG = ROOT / "runs" / "rl" / "learning_curve.png"

st.set_page_config(page_title="Agente Doom — YOLO + DQN", layout="wide")
st.title("Agente Doom con YOLO + Reinforcement Learning")
st.caption("Proyecto final · Teoría de Aprendizaje de Máquinas · UNAL 2026-1")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuracion")
    modo = st.radio(
        "Politica del agente",
        ["DQN (aprendido)", "Reglas (baseline)"],
        help="DQN: aprende de la experiencia. Reglas: logica fija (baseline).",
    )
    max_steps = st.slider("Pasos por episodio", 100, 2000, 600, 100)
    conf_yolo = st.slider("Confianza YOLO", 0.05, 0.50, 0.15, 0.05)
    st.divider()
    st.subheader("Acerca del proyecto")
    st.markdown(
        "**Percepcion:** YOLOv8n entrenado sobre Doom-Enemy-Detection v4 "
        "(668 imágenes, 10 clases de enemigos).  \n"
        "**Politica:** DQN (MLP 8→128→128→7) entrenado 400 episodios "
        "sobre features YOLO + vida + ammo."
    )
    if CURVE_IMG.exists():
        st.image(str(CURVE_IMG), caption="Curva de aprendizaje DQN", use_column_width=True)

# ── Layout principal ───────────────────────────────────────────────────────────
col_video, col_stats = st.columns([3, 1])

with col_video:
    placeholder_video = st.empty()
    placeholder_status = st.empty()

with col_stats:
    st.subheader("Metricas en vivo")
    m_vida   = st.empty()
    m_ammo   = st.empty()
    m_kills  = st.empty()
    m_reward = st.empty()
    m_fps    = st.empty()
    m_accion = st.empty()
    st.divider()
    iniciar  = st.button("▶ Iniciar episodio", use_container_width=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _overlay_action(frame: np.ndarray, label: str) -> np.ndarray:
    img = frame.copy()
    cv2.putText(img, label, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return img


# ── Loop principal ─────────────────────────────────────────────────────────────
def run_dqn():
    import sys; sys.path.insert(0, str(ROOT))
    from src.env.rl_env import RLEnv
    from src.policy.rl_agent import DQNAgent
    from src.policy.actions import Action
    from src.perception.visualization import draw_detections
    from src.utils.metrics import FPSCounter

    if not CKPT_RL.exists():
        st.error(f"Falta el modelo RL en {CKPT_RL}. Entrena con scripts/train_rl.py")
        return

    env   = RLEnv(WEIGHTS, SCENARIO, frame_skip=4, conf=conf_yolo, window_visible=False)
    agent = DQNAgent(env.state_dim, env.n_actions)
    agent.load(CKPT_RL)
    fps = FPSCounter()
    total_reward = 0.0

    try:
        state = env.reset()
        for step in range(max_steps):
            action = agent.act(state, greedy=True)
            state, reward, done, info = env.step(action)
            total_reward += reward

            data = env.last_overlay_data
            if data is not None:
                frame, result = data
                overlay = draw_detections(frame, result)
                overlay = _overlay_action(overlay, f"DQN: {Action(action).name}")
                placeholder_video.image(overlay, channels="RGB", use_container_width=True)

            m_vida.metric("Vida",   int(info.get("vida",  0)))
            m_ammo.metric("Ammo",   int(info.get("ammo",  0)))
            m_kills.metric("Kills", int(info.get("kills", 0)))
            m_reward.metric("Reward", f"{total_reward:.1f}")
            m_fps.metric("FPS", f"{fps.tick():.1f}")
            m_accion.info(f"Accion: **{Action(action).name}**")
            placeholder_status.caption(f"Paso {step+1}/{max_steps}")

            if done:
                placeholder_status.success(f"Episodio terminado en {step+1} pasos | Reward: {total_reward:.1f}")
                break
    finally:
        env.close()


def run_rules():
    import sys; sys.path.insert(0, str(ROOT))
    from src.agent.runner import AgentRunner
    from src.perception.visualization import draw_detections
    from src.utils.metrics import FPSCounter

    if not WEIGHTS.exists():
        st.error(f"Falta el modelo YOLO en {WEIGHTS}.")
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
            overlay = _overlay_action(overlay, f"Reglas: {tick['action'].name}")
            placeholder_video.image(overlay, channels="RGB", use_container_width=True)
            total_reward += tick["reward"]

            m_vida.metric("Vida",   int(tick["info"]["vida"]))
            m_ammo.metric("Ammo",   int(tick["info"]["ammo"]))
            m_kills.metric("Kills", int(tick["info"].get("kills", 0)))
            m_reward.metric("Reward", f"{total_reward:.1f}")
            m_fps.metric("FPS", f"{fps.tick():.1f}")
            m_accion.info(f"Accion: **{tick['action'].name}**")
    finally:
        agent.close()


if iniciar:
    if modo == "DQN (aprendido)":
        run_dqn()
    else:
        run_rules()
