import sys
import copy
import bittensor as bt
import asyncio
from abc import ABC, abstractmethod
from config import check_config, get_config
import argparse
from template.protocol import Dummy
from typing import Tuple
import traceback
import threading
import time
loop = asyncio.get_event_loop()
class Miner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        # Initialize logging
        bt.logging.info("Initializing Miner")

        # Merge provided config with default config
        self.config = copy.deepcopy(config or self.get_default_config())
        bt.logging.info(f"Configurations: {self.config}")

        # Initialize necessary Bittensor components
        self.initialize_wallet(wallet)
        self.initialize_subtensor(subtensor)
        self.initialize_metagraph()
        self.check_wallet_registration()

        # Initialise Axon
        self.initialize_axon(axon)

        # List of function pairs: each pair consists of a forward function and its corresponding blacklist function.
        axon_function_pairs = [
            (self._dummy, self._filter_dummy),
        ]

        self.initialize_axon_connections(axon_function_pairs)
        bt.logging.info(f"Miner initialized with UID: {self.subnet_id}")
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: dict = {}
          
    @abstractmethod
    def config(self) -> bt.config:
        ...

    def get_default_config(self):
        # Load and return the default configuration
        base_config = copy.deepcopy(get_config())
        self.config = self.config()
        self.config.merge(base_config)
        check_config(Miner, self.config)
        return base_config

    def initialize_wallet(self, wallet):
        # Initialize the wallet
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

    def initialize_subtensor(self, subtensor):
        # Initialize subtensor connection
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

    def initialize_metagraph(self):
        # Retrieve and initialize the metagraph
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

    def check_wallet_registration(self):
        # Check if the wallet is registered on the metagraph
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                "Your wallet is not registered. Please register and try again."
            )
            sys.exit()
        self.subnet_id = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

    def initialize_axon(self, axon):
        # Initialize the axon for request processing
        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)
        bt.logging.info(f"Axon: {self.axon}")

    def initialize_axon_connections(self, axon_function_pairs):
        bt.logging.info("Initializing Axon with forward and blacklist functions.")

        # Attaching each pair of functions to the Axon. This includes a forward function and its blacklist counterpart.
        for forward_function, blacklist_function in axon_function_pairs:
            self.axon.attach(forward_fn=forward_function, blacklist_fn=blacklist_function)

        bt.logging.info(f"Axon successfully initialized: {self.axon}")

    def base_blacklist(self, synapse) -> Tuple[bool, str]:
        try:
            hotkey = synapse.dendrite.hotkey
            synapse_type = type(synapse).__name__
            return False, f"accepting {synapse_type} request from {hotkey}"

        except Exception:
            bt.logging.error(f"errror in blacklist {traceback.format_exc()}")

    def _filter_dummy(self, synapse: Dummy) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist
        
    def _dummy(self, synapse: Dummy) -> Dummy:
        return self.dummy(synapse)

    def run_in_background_thread(self) -> None:
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self) -> None:
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            if self.thread.is_alive():
                bt.logging.warning("Thread is still alive after join timeout.")
            self.is_running = False
            bt.logging.debug("Stopped")

    def run(self):
        if not self.subtensor.is_hotkey_registered(netuid=self.config.netuid, hotkey_ss58=self.wallet.hotkey.ss58_address):
            bt.logging.error(f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}. "
                             f"Please register the hotkey using `btcli s register --netuid {self.config.netuid}` before trying again")
            raise RuntimeError("Hotkey not registered")
        
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.chain_endpoint} "
                        f"with netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info("Starting main loop")
        bt.logging.success('\x1b[6;30;42m' + 'Your miner is now active and ready to receive requests.' + '\x1b[0m')
        try:
            self.main_loop()
        except KeyboardInterrupt:
            bt.logging.success("Miner killed by keyboard interrupt.")
        except Exception as e:
            bt.logging.error(f"An error occurred: {e}")
        finally:
            self.clean_up()

    def main_loop(self):
        step = 0
        while not self.should_exit:
            self.wait_for_next_epoch()
            self.update_metagraph(step)
            # --- Set weights logic here if applicable
            step += 1

    def wait_for_next_epoch(self):
        current_block = self.subtensor.get_current_block()
        while current_block - self.last_epoch_block < self.config.miner.blocks_per_epoch:
            time.sleep(1)
            current_block = self.subtensor.get_current_block()
            if self.should_exit:
                break
        self.last_epoch_block = current_block

    def update_metagraph(self, step):
        metagraph = self.subtensor.metagraph(netuid=self.config.netuid, lite=True, block=self.last_epoch_block)
        log = (f"Step:{step} | Block:{metagraph.block.item()} | "
               f"Stake:{metagraph.S[self.subnet_id]} | Rank:{metagraph.R[self.subnet_id]} | "
               f"Trust:{metagraph.T[self.subnet_id]} | Consensus:{metagraph.C[self.subnet_id]} | "
               f"Incentive:{metagraph.I[self.subnet_id]} | Emission:{metagraph.E[self.subnet_id]}")
        bt.logging.info(log)

    def clean_up(self):
        self.axon.stop()
        bt.logging.info("Clean up and shutdown")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()

class TemplateMiner(Miner):
    def config(self) -> bt.config:
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        pass

def main():
    try:
        with TemplateMiner() as miner:
            loop = asyncio.get_event_loop()
            loop.run_forever()
    except KeyboardInterrupt:
        bt.logging.debug("Miner stopped by user.")
    except Exception as e:
        bt.logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    main()