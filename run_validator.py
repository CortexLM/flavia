import argparse
import time
from flavia.utils.autoupdater import AutoUpdater
from subprocess import Popen, PIPE, run
import sys
import subprocess

class SenseProcessManager:
    def __init__(self):
        pass

    @staticmethod
    def update_and_start(process_name, interval=60):
        updater = AutoUpdater()
        updater.check_update()

        # Initialize an empty list to store the arguments
        arguments = []

        # Flag to indicate whether we should skip the next argument (the value after "--process_name")
        skip_next = False

        for arg in sys.argv[1:]:
            if skip_next:
                # Skip the current argument if the flag is set
                skip_next = False
            elif arg == "--process_name":
                # If we encounter "--process_name", set the flag to skip the next argument
                skip_next = True
            else:
                # Otherwise, add the argument to the list of arguments
                arguments.append(arg)
        pm2_command = f"pm2 -f start --interpreter python3 neurons/validator.py --name {process_name}"
        if arguments:
            pm2_command += f" -- {' '.join(arguments)}"
        run(pm2_command, shell=True, check=True)

    @staticmethod
    def check_for_updates(process_name, interval=60):
        updater = AutoUpdater()
        while True:
            time.sleep(interval)
            if updater.check_update():
                pm2_command_stop = f"pm2 stop {process_name}"
                process = Popen(pm2_command_stop, shell=True, stdout=PIPE, stderr=PIPE)
                process.communicate()
                pm2_command_start = f"pm2 start {process_name}"
                process = Popen(pm2_command_start, shell=True, stdout=PIPE, stderr=PIPE)
                process.communicate()

                print("PM2 process successfully restarted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Update Script with AutoUpdater and PM2.")
    parser.add_argument("--process_name", required=True, help="Name of the PM2 process to start.")
    args, unknown = parser.parse_known_args()
    updater_manager = SenseProcessManager()
    updater_manager.update_and_start(args.process_name)
    updater_manager.check_for_updates(args.process_name)
