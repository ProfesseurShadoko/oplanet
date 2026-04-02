
# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
Message.go_root()
import os, sys, subprocess
sys.path.append(os.getcwd())
import time

# ------------------ #
# !-- Subprocess --! #
# ------------------ #

command = "python -m oplanet.alias_searcher"

while True:
    try:
        subprocess.run(command, shell=True, check=True)
        break  # Exit the loop if the command runs successfully
    except subprocess.CalledProcessError as e:
        Message(f"Command failed with error: {e}. Retrying after 20s...", "!")
        for i in ProgressBar(range(100)):
            time.sleep(0.2)