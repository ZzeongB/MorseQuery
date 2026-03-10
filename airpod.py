import argparse
import subprocess
import textwrap
import time

import pyautogui as pg

# Use below command to get the coordinates of the transparency and anc buttons in the sound menu.
# python -c "import pyautogui as p,time; time.sleep(3); print(p.position())"

# 352, 411/ 389, 437

POS = {
    "transparency": (4680, 365),
    "anc": (4691, 403),  # -270 / 2237 / 4777
}

# POS = {
#     "transparency": (1174, 321),
#     "anc": (1132, 378),  # -270 / 2237 / 4777
# }


GET_FRONT_APP = textwrap.dedent(
    r"""
tell application "System Events"
  return name of first process whose frontmost is true
end tell
"""
)

OPEN_SOUND = textwrap.dedent(
    r"""
tell application "System Events"
  tell process "Control Center"
    set frontmost to true
    click (first menu bar item of menu bar 1 whose description is "Sound")
  end tell
end tell
"""
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["transparency", "anc"])
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument(
        "--return-app", default="", help="explicit app name to return focus to"
    )
    parser.add_argument(
        "--no-return-click",
        action="store_true",
        help="do not click original mouse position on return",
    )
    args = parser.parse_args()

    original_pos = pg.position()
    front_app = (
        args.return_app.strip()
        or subprocess.check_output(
            ["osascript", "-e", GET_FRONT_APP], text=True
        ).strip()
    )

    # Move focus to the target display/space first.
    pg.click(*POS[args.mode])
    time.sleep(0.05)

    subprocess.check_output(["osascript", "-e", OPEN_SOUND], text=True)
    time.sleep(args.delay)
    pg.click(*POS[args.mode])

    subprocess.check_output(
        ["osascript", "-e", f'tell application "{front_app}" to activate'], text=True
    )
    if not args.no_return_click:
        pg.click(original_pos.x, original_pos.y)

    print(f"clicked {args.mode} at {POS[args.mode]} and returned to {front_app}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
