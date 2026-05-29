from pathlib import Path
import vizdoom as vzd


class DoomEnv:
    def __init__(self, scenario_cfg: Path, window_visible: bool = False):
        self.game = vzd.DoomGame()
        self.game.load_config(str(scenario_cfg))
        self.game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
        self.game.set_window_visible(window_visible)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()

    def reset(self):
        self.game.new_episode()
        return self._frame()

    def step(self, action_vector: list[int]):
        reward = self.game.make_action(action_vector)
        done = self.game.is_episode_finished()
        frame = None if done else self._frame()
        state = self.game.get_state()
        info = {
            "vida": self.game.get_game_variable(vzd.GameVariable.HEALTH) if state else 0,
            "ammo": self.game.get_game_variable(vzd.GameVariable.AMMO2) if state else 0,
        }
        return frame, reward, done, info

    def _frame(self):
        state = self.game.get_state()
        if state is None:
            return None
        return state.screen_buffer.transpose(1, 2, 0)

    def close(self):
        self.game.close()
