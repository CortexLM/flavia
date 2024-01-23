import sys
import subprocess
import os
path = os.path.dirname(os.path.realpath(__file__))
# Predefined path of the script to run
SCRIPT_PATH = f"{path}/../src/flavia/neurons/validator/validator.py"

def run_script(args):
    # Build the complete command with arguments
    command = [sys.executable, SCRIPT_PATH] + args

    # Execute the script
    subprocess.run(command)

