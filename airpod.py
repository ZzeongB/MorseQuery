import argparse
import subprocess
import textwrap
import time

import pyautogui as pg

POS = {
    "transparency": (2220, 352),
    "anc": (2237, 411),
}

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["transparency", "anc"])
args = parser.parse_args()

script = textwrap.dedent(r"""
tell application "System Events"
  tell process "Control Center"
    set frontmost to true
    click (first menu bar item of menu bar 1 whose description is "Sound")
  end tell
end tell
""")

print(subprocess.check_output(["osascript", "-e", script], text=True))
time.sleep(0.25)  # Sound 패널 열릴 시간
pg.click(*POS[args.mode])
print(f"clicked {args.mode} at {POS[args.mode]}")
