import sys
from pathlib import Path

import cv2
import vizdoom as vzd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "raw-capturas"


def main(n_frames: int = 200, scenario_cfg: str | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    game = vzd.DoomGame()
    if scenario_cfg:
        game.load_config(scenario_cfg)
    else:
        game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
        game.set_doom_map("map01")
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_window_visible(False)
    game.init()

    saved = 0
    while saved < n_frames:
        game.new_episode()
        while not game.is_episode_finished() and saved < n_frames:
            state = game.get_state()
            if state is not None:
                frame = state.screen_buffer.transpose(1, 2, 0)
                cv2.imwrite(str(OUT_DIR / f"frame_{saved:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved += 1
            game.make_action([0, 0, 0, 0, 1, 0])
    game.close()
    print(f"Guardadas {saved} capturas en {OUT_DIR}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
