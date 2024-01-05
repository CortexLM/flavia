import time
import copy
import asyncio
import argparse
import threading
import traceback
import io
import bittensor as bt
from config import get_config, check_config
from functools import partial
import base64
from PIL import Image
import torch
from torchvision import transforms
from starlette.types import Send
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Callable, Awaitable
from template.utils.daemon import DaemonClient
import numpy as np
from template.protocol import TextInteractive, TextCompletion, TextToImage, ImageToImage, isOnline, Models, StartModel, ServerInfo, StopModel
bt.debug()
from io import BytesIO
transform_b64_bt = transforms.Compose([
    transforms.Lambda(lambda x: base64.b64decode(x)),
    transforms.Lambda(lambda x: Image.open(io.BytesIO(x))), 
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
class StreamMiner(ABC):
    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        bt.logging.info("starting stream miner")
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.daemon = DaemonClient(base_url="http://127.0.0.1:8000")
        self.config.merge(base_config)
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain, facilitating interaction with the Bittensor blockchain.
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint} with config:"
        )

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # The axon handles request processing, allowing validators to send this process requests.
        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)
        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to axon.")
        self.axon.attach(
            forward_fn=self._interactive,
        ).attach(
            forward_fn=self._completion,
        ).attach(
            forward_fn=self._text2image,
        ).attach(
            forward_fn=self._image2image,
        ).attach(
            forward_fn=self._is_online,
        ).attach(
            forward_fn=self._models,
        ).attach(
            forward_fn=self._server_info,
        ).attach(
            forward_fn=self._start_model,
        ).attach(
            forward_fn=self._stop_model,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.request_timestamps: Dict = {}

    def config(self) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...
    async def _is_online(self, synapse: isOnline) -> isOnline:
        return True;

    async def _interactive(self, synapse: TextInteractive) -> TextInteractive:
        bt.logging.info(f"started processing for synapse {synapse}")
        
        
        async def _prompt(synapse, send: Send):
            try:
                model = synapse.model
                prompt = synapse.prompt
                bt.logging.info(synapse)
                bt.logging.info(f"question is {prompt} with model {model}")
                buffer = []
                N=1
                for chunk in self.daemon.send_text_generation_interactive(model_name = synapse.model, prompt = synapse.prompt, temperature = synapse.temperature, repetition_penalty = synapse.repetition_penalty, top_p = synapse.top_p, top_k = synapse.top_k, max_tokens = synapse.max_tokens):
                    token = chunk['text']
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
                        bt.logging.info(f"Streamed tokens: {joined_buffer}")
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

    async def _completion(self, synapse: TextCompletion) -> TextCompletion:
        bt.logging.info(f"started processing for synapse {synapse}")
        
        
        async def _prompt(synapse, send: Send):
            try:
                model = synapse.model
                messages = synapse.messages
                bt.logging.info(synapse)
                bt.logging.info(f"question is {messages} with model {model}")
                buffer = []
                N=1
                for chunk in self.daemon.send_text_generation_completions(model = synapse.model, messages = synapse.messages, temperature = synapse.temperature, repetition_penalty = synapse.repetition_penalty, top_p = synapse.top_p,max_tokens = synapse.max_tokens):
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
                        bt.logging.info(f"Streamed tokens: {joined_buffer}")
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

    async def _text2image(self, synapse: TextToImage) -> TextToImage:
        bt.logging.debug(synapse)
        r_output = self.daemon.send_text_to_image_request(model=synapse.model, prompt=synapse.prompt, height=synapse.height, width=synapse.width, num_inference_steps=synapse.num_inference_steps, seed=synapse.seed, batch_size=synapse.batch_size, refiner=synapse.refiner)
        tensor_images = [bt.Tensor.serialize( transform_b64_bt(base64_image) ) for base64_image in r_output['images']]

        synapse.output = tensor_images
        return synapse
    async def _image2image(self, synapse: ImageToImage) -> ImageToImage:
        tensor = bt.Tensor.deserialize(synapse.image);
        pil_image = tensor_to_pil(tensor)
        base64_image = pil_to_base64(pil_image)
        r_output = self.daemon.send_image_to_image_request(model=synapse.model, image=base64_image, prompt=synapse.prompt, height=synapse.height, width=synapse.width, strength=synapse.strength, seed=synapse.seed, batch_size=synapse.batch_size)
        
        tensor_images = [bt.Tensor.serialize( transform_b64_bt(base64_image) ) for base64_image in r_output['images']]

        synapse.output = tensor_images
        return synapse

    async def _models(self, synapse: Models) -> Models:  
        pass
    async def _server_info(self, synapse: ServerInfo) -> ServerInfo:
        pass

    async def _start_model(self, synapse: StartModel) -> StartModel:
        pass

    async def _stop_model(self, synapse: StopModel) -> StopModel:
        pass

    @abstractmethod
    def interactive(self, synapse: TextInteractive) -> TextInteractive:
        pass

    @abstractmethod
    def completion(self, synapse: TextCompletion) -> TextCompletion:
        pass    

    @abstractmethod
    def text2image(self, synapse: TextToImage) -> TextToImage:
        pass  

    @abstractmethod
    def image2image(self, synapse: ImageToImage) -> ImageToImage:
        pass  

    @abstractmethod
    def is_online(self, synapse: isOnline) -> isOnline:
        pass  

    @abstractmethod
    def models(self, synapse: Models) -> Models:
        pass  

    @abstractmethod
    def server_info(self, synapse: ServerInfo) -> ServerInfo:
        pass  

    @abstractmethod
    def start_model(self, synapse: StartModel) -> StartModel:
        pass

    @abstractmethod
    def stop_model(self, synapse: StopModel) -> StopModel:
        pass

    def run(self):
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
                f"Please register the hotkey using `btcli s register --netuid 17` before trying again"
            )
            exit()
        bt.logging.info(
            f"Serving axon {TextInteractive} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()
        self.last_epoch_block = self.subtensor.get_current_block()
        bt.logging.info(f"Miner starting at block: {self.last_epoch_block}")
        bt.logging.info(f"Starting main loop")
        step = 0
        try:
            while not self.should_exit:
                start_epoch = time.time()

                # --- Wait until the next epoch.
                current_block = self.subtensor.get_current_block()
                while (
                    current_block - self.last_epoch_block
                    < self.config.miner.blocks_per_epoch
                ):
                    # --- Wait for the next block.
                    time.sleep(1)
                    current_block = self.subtensor.get_current_block()
                    # --- Check if we should exit.
                    if self.should_exit:
                        break

                # --- Update the metagraph with the latest network state.
                self.last_epoch_block = self.subtensor.get_current_block()

                metagraph = self.subtensor.metagraph(
                    netuid=self.config.netuid,
                    lite=True,
                    block=self.last_epoch_block,
                )
                log = (
                    f"Step:{step} | "
                    f"Block:{metagraph.block.item()} | "
                    f"Stake:{metagraph.S[self.my_subnet_uid]} | "
                    f"Rank:{metagraph.R[self.my_subnet_uid]} | "
                    f"Trust:{metagraph.T[self.my_subnet_uid]} | "
                    f"Consensus:{metagraph.C[self.my_subnet_uid] } | "
                    f"Incentive:{metagraph.I[self.my_subnet_uid]} | "
                    f"Emission:{metagraph.E[self.my_subnet_uid]}"
                )
                bt.logging.info(log)

                # --- Set weights.
                if not self.config.miner.no_set_weights:
                    pass
                step += 1

        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def run_in_background_thread(self):
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        if self.is_running:
            bt.logging.debug("Stopping miner in the background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_run_thread()


class StreamingTemplateMiner(StreamMiner):
    def add_args(cls, parser: argparse.ArgumentParser):
        pass

    def interactive(self, synapse: TextInteractive) -> TextInteractive:
        ...

    def completion(self, synapse: TextCompletion)  -> TextCompletion:
        ...

    def text2image(self, synapse: TextToImage)  -> TextToImage:
        ...

    def image2image(self, synapse: ImageToImage)  -> ImageToImage:
        ...

    def is_online(self, synapse: ImageToImage)  -> ImageToImage:
        ...

    def models(self, synapse: ImageToImage)  -> ImageToImage:
        ...

    def server_info(self, synapse: ImageToImage)  -> ImageToImage:
        ...

    def start_model(self, synapse: StartModel)  -> StartModel:
        ...

    def stop_model(self, synapse: StopModel)  -> StopModel:
        ...

if __name__ == "__main__":
    with StreamingTemplateMiner():
        while True:
            time.sleep(1)