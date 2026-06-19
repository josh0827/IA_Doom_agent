import os
from pathlib import Path
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario_cfg: Path, window_visible: bool = False, skill: int | None = None,
                 cheats: bool = False):
        self.game = vzd.DoomGame()
        self.game.load_config(str(scenario_cfg))
        # cheats: habilita la consola de ZDoom (god, give, summon) para la demo de
        # sala controlada, donde spawneamos a mano solo enemigos detectables.
        self._cheats = cheats
        if cheats:
            self.game.add_game_args("+sv_cheats 1")
        # Inyecta el .wad buscando en este orden:
        # 1. <scenarios_dir>/<stem>.wad  (escenarios de ViZDoom, e.g. deadly_corridor)
        # 2. <mismo directorio que el cfg>/<stem>.wad  (WADs propios, e.g. freedoom1)
        # 3. <mismo directorio que el cfg>/freedoom1.wad  (alias para cfg "freedoom")
        scenarios_dir = Path(vzd.scenarios_path)
        cfg_dir = scenario_cfg.parent
        candidates = [
            scenarios_dir / f"{scenario_cfg.stem}.wad",
            cfg_dir / f"{scenario_cfg.stem}.wad",
            cfg_dir / "freedoom1.wad",
        ]
        for wad in candidates:
            if wad.exists():
                self.game.set_doom_scenario_path(str(wad))
                break
        if skill is not None:
            self.game.set_doom_skill(skill)
        self.game.set_window_visible(window_visible)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        # Renderizado por GPU (OpenGL): acelera frames con depth buffer activo.
        # En servidores sin pantalla (Kaggle/Colab) OpenGL falla -> con DOOM_HEADLESS
        # se omite y ViZDoom usa el software renderer headless.
        if not os.environ.get("DOOM_HEADLESS"):
            self.game.add_game_args("+vid_renderer opengl +gl_multisample 0")
        self.game.init()
        # POSITION_X disponible solo si el cfg lo declara (deadly_corridor): permite
        # medir avance real para la recompensa de progreso potencial.
        self._has_posx = vzd.GameVariable.POSITION_X in self.game.get_available_game_variables()
        self._last_info = {"vida": 100, "ammo": 0, "kills": 0, "pos_x": 0.0}
        self._last_depth = None

    def reset(self):
        self._last_info = {"vida": 100, "ammo": 0, "kills": 0}
        self.game.new_episode()
        return self._frame()

    def step(self, action_vector: list[int], tics: int = 1):
        # tics > 1 aplica la misma accion varios frames (frame-skip).
        reward = self.game.make_action(action_vector, tics)
        done = self.game.is_episode_finished()
        frame = None if done else self._frame()
        state = self.game.get_state()
        if state:
            self._last_info = {
                "vida":  self.game.get_game_variable(vzd.GameVariable.HEALTH),
                "ammo":  self.game.get_game_variable(vzd.GameVariable.AMMO2),
                "kills": self.game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                "pos_x": (self.game.get_game_variable(vzd.GameVariable.POSITION_X)
                          if self._has_posx else 0.0),
            }
        # Al morir (state=None) devuelve el ultimo valor conocido para no perder kills.
        info = dict(self._last_info)
        dead = self.game.is_player_dead()
        info["dead"] = dead  # fiable aunque state sea None
        # Completo el nivel = termino vivo y ANTES del timeout (alcanzo el chaleco).
        # Si agota el tiempo, get_episode_time() == get_episode_timeout().
        info["completed"] = bool(
            done and not dead
            and self.game.get_episode_time() < self.game.get_episode_timeout()
        )
        return frame, reward, done, info

    def send(self, command: str):
        """Envia un comando de consola de ZDoom (requiere cheats=True). Ej: 'summon Demon'."""
        self.game.send_game_command(command)

    def _frame(self):
        state = self.game.get_state()
        if state is None:
            self._last_depth = None
            return None
        self._last_depth = state.depth_buffer  # (H, W) float32, distancia en unidades Doom
        return state.screen_buffer  # ViZDoom 1.3+ devuelve (H, W, 3) directamente

    @property
    def depth_buffer(self):
        return self._last_depth

    def close(self):
        self.game.close()
