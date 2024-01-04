
from utils.logging import logging
from utils.model import ModelManager
import utils.system as system
import time
from utils.fastapi import DaemonAPI
def main():
    logging.warning("The daemon server must not be on the same server as the miner/validator.")
    time.sleep(2)
    logging.info("Initializing Flavia Subnet [ρ] Daemon...")
    system.display_system_info()
    model = ModelManager()
    api = DaemonAPI(model=model)
    api.run()

if __name__ == "__main__":
    main()