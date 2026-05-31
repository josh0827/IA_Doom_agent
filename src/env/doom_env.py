from pathlib import Path
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario_cfg: Path, window_visible: bool = False):
        self.game = vzd.DoomGame()
        self.game.load_config(str(scenario_cfg))
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
        self.game.set_window_visible(window_visible)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()
        self._last_info = {"vida": 100, "ammo": 0, "kills": 0}

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
            }
        # Al morir (state=None) devuelve el ultimo valor conocido para no perder kills.
        info = dict(self._last_info)
        info["dead"] = self.game.is_player_dead()  # fiable aunque state sea None
        return frame, reward, done, info

    def _frame(self):
        state = self.game.get_state()
        if state is None:
            return None
        return state.screen_buffer  # ViZDoom 1.3+ devuelve (H, W, 3) directamente

    def close(self):
        self.game.close()
