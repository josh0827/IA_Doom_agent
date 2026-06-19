from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.paths import detector_weights
try:
    WEIGHTS = detector_weights()
except FileNotFoundError:
    WEIGHTS = ROOT / "runs" / "doom-v1" / "weights" / "best.pt"  # fallback para mostrar error claro

# Escenarios disponibles: cada uno con su politica entrenada y su curva.
#  - Pasillo (deadly_corridor): avanzar hasta el chaleco; politica dqn.pt.
#  - Sala  (defend_the_center): torreta que gira y dispara; politica dqn_room.pt
#    (modo torreta = se prohiben las acciones de avanzar en inferencia).
ESCENARIOS = {
    "Pasillo (deadly_corridor)": {
        "cfg":   ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg",
        "ckpt":  ROOT / "runs" / "rl" / "dqn.pt",
        "curva": ROOT / "runs" / "rl" / "learning_curve.png",
        "torreta": False,
    },
    "Sala abierta (defend_the_center)": {
        "cfg":   ROOT / "src" / "env" / "scenarios" / "defend_the_center.cfg",
        "ckpt":  ROOT / "runs" / "rl" / "dqn_room.pt",
        "curva": ROOT / "runs" / "rl" / "learning_curve_room.png",
        "torreta": True,
    },
    # Demo controlada: arena cerrada (mapa health_gathering) donde spawneamos a
    # mano SOLO enemigos de alta detectabilidad (pinky + zombieman). god + give
    # all para una demostracion visual limpia de percepcion + disparo.
    "Sala demo (solo pinky + zombieman)": {
        "cfg":   ROOT / "src" / "env" / "scenarios" / "health_gathering.cfg",
        "ckpt":  ROOT / "runs" / "rl" / "dqn_room.pt",
        "curva": ROOT / "runs" / "rl" / "learning_curve_room.png",
        "torreta": True,
        "summon": ["Demon", "Zombieman"],
    },
}

st.set_page_config(page_title="Agente Doom — YOLO + DQN", layout="wide")
st.title("Agente Doom con YOLO + Reinforcement Learning")
st.caption("Proyecto final · Teoría de Aprendizaje de Máquinas · UNAL 2026-1")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuracion")
    escenario_nombre = st.radio(
        "Escenario",
        list(ESCENARIOS.keys()),
        help="Pasillo: avanzar a la meta. Sala: sobrevivir girando (torreta).",
    )
    modo = st.radio(
        "Politica del agente",
        ["DQN (aprendido)", "Reglas (baseline)"],
        help="DQN: aprende de la experiencia. Reglas: logica fija (baseline).",
    )
    max_steps = st.slider("Pasos por episodio", 100, 2000, 600, 100)
    conf_yolo = st.slider("Confianza YOLO", 0.05, 0.70, 0.40, 0.01)
    st.divider()
    st.subheader("Acerca del proyecto")
    st.markdown(
        "**Percepcion:** YOLOv8s entrenado sobre un dataset in-domain "
        "auto-etiquetado desde ViZDoom (2213 imágenes; 4 clases fiables). "
        "mAP@0.5 = 0.908.  \n"
        "**Politica:** Double DQN con Dueling + PER + n-step "
        "(estado 39 dims = 13 features × 3 frames, cabezas V/A).  \n"
        "**Acciones:** 13 (strafe, giro+disparo, kiting). En la sala se "
        "enmascaran las de avanzar (torreta)."
    )

ESC = ESCENARIOS[escenario_nombre]
SCENARIO  = ESC["cfg"]
CKPT_RL   = ESC["ckpt"]
CURVE_IMG = ESC["curva"]
TORRETA   = ESC["torreta"]
SUMMON    = ESC.get("summon")  # lista de enemigos a spawnear (None = escenario nativo)

with st.sidebar:
    if CURVE_IMG.exists():
        st.image(str(CURVE_IMG), caption=f"Curva de aprendizaje — {escenario_nombre}",
                 use_column_width=True)

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
    from src.env.frame_stack import FrameStack
    from src.env.rl_env import RLEnv
    from src.policy.rl_agent import DQNAgent
    from src.policy.actions import Action
    from src.perception.visualization import draw_detections
    from src.utils.metrics import FPSCounter

    if not CKPT_RL.exists():
        st.error(f"Falta el modelo RL en {CKPT_RL}. Entrena con scripts/train_rl.py")
        return

    env   = FrameStack(RLEnv(WEIGHTS, SCENARIO, frame_skip=2, conf=conf_yolo,
                             window_visible=False, cheats=bool(SUMMON)), n_frames=3)
    agent = DQNAgent(env.state_dim, env.n_actions)
    agent.load(CKPT_RL)
    # Modo torreta (sala): se prohibe avanzar enmascarando esas acciones.
    forbidden = ({int(Action.MOVE_FORWARD), int(Action.FORWARD_ATTACK)}
                 if TORRETA else None)
    fps = FPSCounter()
    total_reward = 0.0

    def poblar_arena(n=3):
        """Spawnea enemigos detectables alrededor del jugador (demo controlada)."""
        for actor in SUMMON:
            for _ in range(n):
                env.env.send(f"summon {actor}")

    def equipar():
        """god + escopeta equipada (slot 3, el arma con que entreno la politica)
        + municion. 'give all' NO autoselecciona arma: hay que forzar el slot."""
        env.env.send("god")
        env.env.send("give all")
        env.env.send("slot 3")     # escopeta: arma de la politica de sala
        env.env.send("give ammo")

    try:
        state = env.reset()
        if SUMMON:
            equipar()
            poblar_arena(4)
        for step in range(max_steps):
            if SUMMON:
                # Municion infinita: rellena seguido para que nunca se quede sin balas.
                if step % 15 == 0:
                    env.env.send("give ammo")
                # Re-poblar la arena para que siempre haya enemigos a la vista.
                if step > 0 and step % 45 == 0:
                    poblar_arena(2)
            action = agent.act(state, greedy=True, forbidden=forbidden)
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
