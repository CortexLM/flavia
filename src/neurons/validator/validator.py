import sys
import bittensor as bt
import argparse
from pathlib import Path
from src.neurons.validator.validations.dummy import DummyValidator
from src.neurons.validator.utils.weights import Weights
import asyncio 

class BittensorValidator:
    def __init__(self):
        self.metagraph = None
        # Classes Validators
        self.dummy_validator = None
        self.config = self.get_config()
        print(self.config)
        self.wallet, self.subtensor, self.dendrite, self.validator_uid = self.setup_validator(self.config)
        self.setup_validation_components()
        self.loop = asyncio.get_event_loop()

    async def run_forever(self):
        while True:
            await asyncio.sleep(60)

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, default=62)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        _args = parser.parse_args()
        full_path = Path(
            f"{config.logging.logging_dir}/{config.wallet.name}/{config.wallet.hotkey}/netuid{config.netuid}/validator"
        ).expanduser()
        config.full_path = str(full_path)
        full_path.mkdir(parents=True, exist_ok=True)
        print(config)
        return config
    
    async def run(self):
        self.weights = Weights(
        self.loop, self.dendrite, self.subtensor, self.config, self.wallet, self.dummy_validator)
        await self.run_forever()
    def setup_validator(self, config):
        bt.logging.info(f"Running validator for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint}")
        wallet = bt.wallet(config=config)
        subtensor = bt.subtensor(config=config)
        self.metagraph = subtensor.metagraph(config.netuid)
        dendrite = bt.dendrite(wallet=wallet)
        try:
            validator_uid = self.metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        except ValueError:
            bt.logging.error(
                f"Your validator: {wallet} is not registered to chain connection: {subtensor}. "
                f"Run btcli register --netuid {config.netuid} and try again."
            )
            sys.exit()
        return wallet, subtensor, dendrite, validator_uid

    def setup_validation_components(self):
        validator_config = {
            "dendrite": self.dendrite,
            "config": self.config,
            "subtensor": self.subtensor,
            "wallet": self.wallet
        }
        self.dummy_validator = DummyValidator(**validator_config)
        bt.logging.info("Validators initialized successfully.")

if __name__ == "__main__":
    validator = BittensorValidator()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(validator.run())
