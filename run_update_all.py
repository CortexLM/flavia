import argparse
import time
from flavia.autoupdater import AutoUpdater
from subprocess import Popen, PIPE, run
import sys
import subprocess

class FlaviaProcessUpdateManager:
    def __init__(self):
        pass

    @staticmethod
    def check_for_updates(interval=60):
        updater = AutoUpdater()
        while True:
            time.sleep(interval)
            if updater.check_update():
                pm2_command_stop = f"pm2 stop all"
                process = Popen(pm2_command_stop, shell=True, stdout=PIPE, stderr=PIPE)
                process.communicate()
                pm2_command_start = f"pm2 start all"
                process = Popen(pm2_command_start, shell=True, stdout=PIPE, stderr=PIPE)
                process.communicate()

                print("PM2 process successfully restarted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Update Script with AutoUpdater and PM2.")
    updater_manager = FlaviaProcessUpdateManager()
    updater_manager.check_for_updates()