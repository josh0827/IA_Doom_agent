from pathlib import Path
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario_cfg: Path, window_visible: bool = False):
        self.game = vzd.DoomGame()
        self.game.load_config(str(scenario_cfg))
        # Inyecta el .wad oficial de ViZDoom segun el nombre del cfg
        # (p.ej. deadly_corridor.cfg -> deadly_corridor.wad).
        wad = Path(vzd.scenarios_path) / f"{scenario_cfg.stem}.wad"
        if wad.exists():
            self.game.set_doom_scenario_path(str(wad))
        self.game.set_window_visible(window_visible)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()

    def reset(self):
        self.game.new_episode()
        return self._frame()

    def step(self, action_vector: list[int], tics: int = 1):
        # tics > 1 aplica la misma accion varios frames (frame-skip).
        reward = self.game.make_action(action_vector, tics)
        done = self.game.is_episode_finished()
        frame = None if done else self._frame()
        state = self.game.get_state()
        info = {
            "vida": self.game.get_game_variable(vzd.GameVariable.HEALTH) if state else 0,
            "ammo": self.game.get_game_variable(vzd.GameVariable.AMMO2) if state else 0,
            "kills": self.game.get_game_variable(vzd.GameVariable.KILLCOUNT) if state else 0,
        }
        return frame, reward, done, info

    def _frame(self):
        state = self.game.get_state()
        if state is None:
            return None
        return state.screen_buffer  # ViZDoom 1.3+ devuelve (H, W, 3) directamente

    def close(self):
        self.game.close()
