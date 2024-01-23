from io import BytesIO
import sys
import copy
import bittensor as bt
import asyncio
from abc import ABC, abstractmethod
from config import check_config, get_config
import argparse
from src.flavia.neurons.validator.rewards.text2image import calculate_image_timeout
from flavia.protocol import TextCompletion, TextToImage
from typing import Tuple
import traceback
import threading
import time
from starlette.types import Send
from functools import partial
from flavia.sense import SenseClient
from torchvision import transforms
import torch
from PIL import Image
import base64
loop = asyncio.get_event_loop()
bt.debug()
transform_b64_bt = transforms.Compose([
    transforms.Lambda(lambda x: base64.b64decode(x)),
    transforms.Lambda(lambda x: Image.open(BytesIO(x))), 
    transforms.ToTensor() 
])
def tensor_to_pil(tensor_image):
    # Normalize tensor to 0-1 if it's not already
    if torch.max(tensor_image) > 1:
        tensor_image = tensor_image / 255

    # Convert to PIL image
    return Image.fromarray(tensor_image.mul(255).byte().numpy().transpose(1, 2, 0))

# Convert PIL image to base64
def pil_to_base64(pil_image):
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()
class Miner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        # Initialize logging
        bt.logging.info("Initializing Miner")

        # Merge provided config with default config
        self.config = copy.deepcopy(config or self.get_default_config())
        bt.logging.info(f"Configurations: {self.config}")

        self.sense = SenseClient(base_url=self.config.sense.base_url, api_key=self.config.sense.api_key)

        # Initialize necessary Bittensor components
        self.initialize_wallet(wallet)
        self.initialize_subtensor(subtensor)
        self.initialize_metagraph()
        self.check_wallet_registration()

        # Initialise Axon
        self.initialize_axon(axon)

        # List of function pairs: each pair consists of a forward function and its corresponding blacklist function.
        axon_function_pairs = [
            (self._completion, self._filter_completion),
            (self._text2image, self._filter_text2image)
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

    def _filter_text2image(self, synapse: TextToImage) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    def _filter_completion(self, synapse: TextCompletion) -> Tuple[bool, str]:
        blacklist = self.base_blacklist(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist
        
    async def completion(self, synapse: TextCompletion) -> TextCompletion:
        bt.logging.info(f"started processing for synapse {synapse}")
        
        async def _prompt(synapse, send: Send):
            try:
                model = synapse.model
                messages = synapse.messages
                bt.logging.info(synapse)
                bt.logging.info(f"question is {messages} with model {model}")
                buffer = []
                N=1
                async for chunk in self.sense.completion(model = synapse.model, messages = synapse.messages, temperature = synapse.temperature, repetition_penalty = synapse.repetition_penalty, top_p = synapse.top_p,max_tokens = synapse.max_tokens):
                    token = chunk['text']
                    token = token.replace('\n', '<newline>')
                    buffer.append(token)
                    if len(buffer) == N:
                        joined_buffer = "".join(buffer)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        bt.logging.debug(f"Streamed tokens: {joined_buffer}")
                        buffer = []

                if buffer:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )
                    bt.logging.info(f"Streamed tokens: {joined_buffer}")
            except Exception as e:
                 bt.logging.error(f"error in _prompt {e}\n{traceback.format_exc()}")

        token_streamer = partial(_prompt, synapse)
        return synapse.create_streaming_response(token_streamer)

    async def text2image(self, synapse: TextToImage) -> TextToImage:
        start_time = time.time()  # Start timing

        bt.logging.debug(synapse)
        bt.logging.debug(f"timeout for this image : {calculate_image_timeout(synapse.num_inference_steps)}")
        r_output = await self.sense.text2image(model=synapse.model, prompt=synapse.prompt, height=synapse.height, width=synapse.width, num_inference_steps=synapse.num_inference_steps, seed=synapse.seed, batch_size=synapse.batch_size, refiner=synapse.refiner)
        tensor_images = [bt.Tensor.serialize(transform_b64_bt(base64_image)) for base64_image in r_output['images']]

        synapse.output = tensor_images

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        bt.logging.debug(f"Elapsed time for image generation : {elapsed_time:.2f} seconds")  # Print elapsed time

        return synapse 

    async def _text2image(self, synapse: TextToImage) -> TextToImage:
        return await self.text2image(synapse)
    async def _completion(self, synapse: TextCompletion) -> TextCompletion:
        return await self.completion(synapse)

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