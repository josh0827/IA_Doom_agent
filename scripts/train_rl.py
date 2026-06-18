"""Entrena el agente Double DQN (Dueling + PER) sobre Doom con FrameStack.

Mejoras activas:
  - FrameStack (3 frames): contexto temporal sin LSTM
  - PER (Prioritized Experience Replay): foco en transiciones importantes
  - n-step returns (n=3): credito mas preciso para secuencias de acciones
  - TensorBoard: metricas en tiempo real
  - Evaluacion greedy periodica (cada 50 ep): checkpoint del mejor modelo real
  - Curriculum: sube dificultad (doom_skill) segun rendimiento en kills

NOTA: incompatible con checkpoints anteriores (STATE_DIM y N_ACTIONS cambiaron).

Uso:
    cd "proyecto final Agente doom"
    .venv\\Scripts\\activate
    python scripts/train_rl.py                    # 600 ep, GPU automatico
    python scripts/train_rl.py --episodes 800     # mas entrenamiento
    python scripts/train_rl.py --resume           # reanudar desde checkpoint
    python scripts/train_rl.py --window           # ver ventana mientras entrena
    tensorboard --logdir runs/rl/tb               # ver metricas en vivo
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env.frame_stack import FrameStack
from src.env.rl_env import RLEnv
from src.perception.visualization import draw_detections
from src.policy.actions import Action
from src.policy.rl_agent import DQNAgent
from src.utils.paths import detector_weights

SCENARIO = ROOT / "src" / "env" / "scenarios" / "deadly_corridor.cfg"
OUT_DIR  = ROOT / "runs" / "rl"
CKPT     = OUT_DIR / "dqn.pt"
CKPT_LAST = OUT_DIR / "dqn_last.pt"
CURVE    = OUT_DIR / "learning_curve.png"
TB_DIR   = OUT_DIR / "tb"

# Curriculum: kills promedio (20 ep) necesarios para subir de skill
CURRICULUM = {1: 3.0, 2: 5.0}   # skill 1->2 con 3 kills/ep, 2->3 con 5 kills/ep

N_STEP = 3  # n-step returns


class NStepBuffer:
    """Acumula N transiciones y emite el retorno de N pasos."""

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque()

    def push(self, s, a, r, s_next, done):
        self._buf.append((s, a, r, s_next, done))
        if len(self._buf) < self.n:
            return None
        return self._emit()

    def flush(self):
        out = []
        while self._buf:
            out.append(self._emit())
        return out

    def _emit(self):
        s0, a0 = self._buf[0][0], self._buf[0][1]
        # gamma_n acumula gamma^k: al terminar el bucle vale gamma^(pasos sumados),
        # que es exactamente el descuento con el que se debe arrancar (bootstrap)
        # el valor del estado final en el target n-step.
        G, gamma_n = 0.0, 1.0
        sn, dn = self._buf[-1][3], False
        for _, _, r, s_next, done in self._buf:
            G += gamma_n * r
            gamma_n *= self.gamma
            if done:
                sn, dn = s_next, True   # corta en la transicion terminal real
                break
        self._buf.popleft()
        return s0, a0, G, sn, float(dn), gamma_n


def make_env(weights, scenario, frame_skip, skill, window_visible=False):
    return FrameStack(
        RLEnv(weights, scenario, frame_skip=frame_skip,
              window_visible=window_visible, skill=skill),
        n_frames=3,
    )


def evaluate(agent, weights, scenario, frame_skip, skill=1, n_ep=3, max_steps=600,
             forbidden=None) -> tuple[float, float]:
    """Corre N episodios en modo greedy y devuelve (reward_medio, kills_medio)."""
    rewards, kills_list = [], []
    for _ in range(n_ep):
        env = make_env(weights, scenario, frame_skip, skill=skill)
        state = env.reset()
        total, ep_kills = 0.0, 0
        for _ in range(max_steps):
            action = agent.act(state, greedy=True, forbidden=forbidden)
            state, reward, done, info = env.step(action)
            total += reward
            ep_kills = int(info.get("kills", 0))
            if done:
                break
        env.close()
        rewards.append(total)
        kills_list.append(ep_kills)
    return float(np.mean(rewards)), float(np.mean(kills_list))


def plot_curve(rewards, path, window=20):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.35, label="reward por episodio")
    if len(rewards) >= window:
        media = [
            sum(rewards[max(0, i - window + 1):i + 1]) / len(rewards[max(0, i - window + 1):i + 1])
            for i in range(len(rewards))
        ]
        plt.plot(media, color="crimson", linewidth=2, label=f"media movil ({window})")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.title("Curva de aprendizaje — Doom Agent (Double DQN + PER + FrameStack)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=800)
    parser.add_argument("--max-steps",  type=int, default=600)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--device",     type=str, default="cuda", help="cuda | cpu")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--window",     action="store_true")
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--scenario",   type=str, default="deadly_corridor",
                        help="cfg en src/env/scenarios (sin .cfg)")
    parser.add_argument("--no-forward", action="store_true",
                        help="prohibe avanzar (entreno torreta para sala abierta)")
    parser.add_argument("--tag",        type=str, default="",
                        help="sufijo para no pisar pesos/curva (ej: _room)")
    parser.add_argument("--timesteps",  type=int, default=0,
                        help="presupuesto de pasos de entorno (0 = usar --episodes)")
    args = parser.parse_args()
    if args.timesteps > 0:
        args.episodes = 10**9   # corre hasta agotar el presupuesto de pasos

    # Escenario, rutas de salida (por tag) y mascara de acciones.
    global SCENARIO, CKPT, CKPT_LAST, CURVE, TB_DIR
    SCENARIO  = ROOT / "src" / "env" / "scenarios" / f"{args.scenario}.cfg"
    CKPT      = OUT_DIR / f"dqn{args.tag}.pt"
    CKPT_LAST = OUT_DIR / f"dqn_last{args.tag}.pt"
    CURVE     = OUT_DIR / f"learning_curve{args.tag}.png"
    TB_DIR    = OUT_DIR / f"tb{args.tag}"
    forbidden = ({int(Action.MOVE_FORWARD), int(Action.FORWARD_ATTACK)}
                 if args.no_forward else None)
    print(f"Escenario: {args.scenario} | tag='{args.tag}' | "
          f"no_forward={args.no_forward} | curriculum={not args.no_curriculum}")

    # ── Verificacion GPU ──────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ADVERTENCIA: CUDA no disponible, cambiando a CPU.")
        args.device = "cpu"
    if args.device == "cuda":
        gpu_name   = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU : {gpu_name}  ({vram_total:.1f} GB VRAM)")
        print(f"CUDA: {torch.version.cuda}")
        # Optimizaciones GPU
        torch.backends.cudnn.benchmark = True   # autotuning de kernels CUDA
        torch.cuda.set_per_process_memory_fraction(0.85, 0)  # reservar 85% VRAM
        torch.backends.cuda.matmul.allow_tf32 = True          # TF32 en RTX (2x speed)
        print("Optimizaciones CUDA activas: cudnn.benchmark + TF32 + 85% VRAM")
    else:
        print("Dispositivo: CPU")

    weights = detector_weights()
    curr_skill = 1
    env = make_env(weights, SCENARIO, args.frame_skip, curr_skill,
                   window_visible=args.window)
    agent = DQNAgent(env.state_dim, env.n_actions, device=args.device)
    print(f"state_dim={env.state_dim} | n_actions={env.n_actions} | buffer PER")

    if args.resume and CKPT.exists():
        agent.load(CKPT)
        print(f"Reanudando desde {CKPT}")

    # TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(TB_DIR))
        use_tb = True
        print(f"TensorBoard activo → tensorboard --logdir {TB_DIR}")
    except ImportError:
        writer = None
        use_tb = False
        print("TensorBoard no instalado. Instala con: pip install tensorboard")

    rewards_hist: list[float] = []
    kills_hist:   list[float] = []
    recientes  = deque(maxlen=20)
    kills_rec  = deque(maxlen=20)
    mejor_eval = float("-inf")
    nstep_buf  = NStepBuffer(N_STEP, agent.gamma)
    total_steps = 0   # pasos de entorno acumulados (para --timesteps)

    try:
        for ep in range(1, args.episodes + 1):
            state = env.reset()
            total, losses, ep_kills = 0.0, [], 0

            for step in range(args.max_steps):
                action = agent.act(state, forbidden=forbidden)
                next_state, reward, done, info = env.step(action)
                total_steps += 1

                # n-step: acumula y empuja al buffer cuando hay N transiciones
                transition = nstep_buf.push(state, action, reward, next_state, float(done))
                if transition:
                    agent.buffer.push(*transition)

                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total += reward
                ep_kills = int(info.get("kills", 0))

                if args.window:
                    data = env.last_overlay_data
                    if data is not None:
                        frame, result = data
                        overlay = draw_detections(frame, result)
                        cv2.putText(overlay,
                            f"Ep {ep} | skill={curr_skill} | eps={agent.epsilon():.2f} | {Action(action).name}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        cv2.imshow("Entrenando", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt
                if done:
                    break

            # Vaciar el buffer n-step al final del episodio
            for t in nstep_buf.flush():
                agent.buffer.push(*t)

            rewards_hist.append(total)
            kills_hist.append(ep_kills)
            recientes.append(total)
            kills_rec.append(ep_kills)
            media        = sum(recientes) / len(recientes)
            kills_media  = sum(kills_rec) / len(kills_rec)
            loss_str     = f"{sum(losses)/len(losses):.4f}" if losses else "  N/A  "

            prog = (f"pasos {total_steps}/{args.timesteps}" if args.timesteps
                    else f"Ep {ep:4d}/{args.episodes}")
            print(
                f"{prog} | skill={curr_skill} | "
                f"reward={total:7.1f} | media20={media:7.1f} | "
                f"kills={ep_kills} | kills_avg={kills_media:.1f} | "
                f"eps={agent.epsilon():.3f} | loss={loss_str}"
            )

            # TensorBoard
            if use_tb:
                writer.add_scalar("Reward/episodio",   total,       ep)
                writer.add_scalar("Reward/media20",    media,       ep)
                writer.add_scalar("Game/kills",        ep_kills,    ep)
                writer.add_scalar("Game/kills_avg20",  kills_media, ep)
                writer.add_scalar("DQN/epsilon",       agent.epsilon(), ep)
                writer.add_scalar("DQN/skill",         curr_skill,  ep)
                if losses:
                    writer.add_scalar("DQN/loss", sum(losses)/len(losses), ep)
                writer.add_scalar("DQN/lr", agent.scheduler.get_last_lr()[0], ep)

            # Reporte GPU cada 50 episodios
            if ep % 50 == 0 and args.device == "cuda":
                used  = torch.cuda.memory_allocated(0) / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                print(f"  [GPU] VRAM usada: {used:.0f} MB / {total:.0f} MB")

            # Evaluacion greedy periodica
            if ep % 50 == 0:
                eval_reward, eval_kills = evaluate(agent, weights, SCENARIO, args.frame_skip,
                                                   skill=curr_skill, forbidden=forbidden)
                print(f"  [EVAL] reward={eval_reward:.1f} | kills={eval_kills:.1f}")
                if use_tb:
                    writer.add_scalar("Eval/reward", eval_reward, ep)
                    writer.add_scalar("Eval/kills",  eval_kills,  ep)
                if eval_reward > mejor_eval:
                    mejor_eval = eval_reward
                    agent.save(CKPT)
                    print(f"  [CKPT] Nuevo mejor guardado ({eval_reward:.1f})")

            if ep % 20 == 0:
                plot_curve(rewards_hist, CURVE)

            # Curriculum: subir skill si los kills promedio superan el umbral
            if not args.no_curriculum and curr_skill < 3 and len(kills_rec) == 20:
                umbral = CURRICULUM.get(curr_skill, 999)
                if kills_media >= umbral:
                    curr_skill += 1
                    env.close()
                    env = make_env(weights, SCENARIO, args.frame_skip, curr_skill)
                    print(f"  [CURRICULUM] Subiendo a skill {curr_skill} (kills_avg={kills_media:.1f})")

            # Presupuesto por pasos (--timesteps): corta al alcanzarlo
            if args.timesteps and total_steps >= args.timesteps:
                print(f"\nPresupuesto alcanzado: {total_steps} pasos en {ep} episodios.")
                break

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido.")
    finally:
        env.close()
        if args.window:
            cv2.destroyAllWindows()
        agent.save(CKPT_LAST)
        plot_curve(rewards_hist, CURVE)
        if use_tb:
            writer.close()
        print(f"\nGuardado: {CKPT} (mejor eval) | {CKPT_LAST} (ultimo)")
        print(f"Curva: {CURVE}")
        print(f"TensorBoard: tensorboard --logdir {TB_DIR}")


if __name__ == "__main__":
    main()
