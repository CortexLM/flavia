import aiohttp
import bittensor as bt
import json

def add_args(parser):
    # Adds arguments related to Sense configuration to the parser

    # Argument for Sense base URL
    parser.add_argument(
        "--sense.base_url",
        type=str,
        help="Base URL for Sense Daemon",
        default="http://127.0.0.1:8000",
    )

    # Argument for Sense API key
    parser.add_argument(
        "--sense.api_key",
        type=str,
        help="API key for Sense Daemon.",
        default=None,
    )

class SenseClient:
    def __init__(self, base_url='http://127.0.0.1:8000', api_key=None):
        if base_url.endswith('/'):
            # Si elle se termine par '/', supprimez-le en utilisant le slicing
            base_url = base_url[:-1]
        bt.logging.debug(f"Use Sense Server {base_url}")
        self.base_url = base_url
        
        self.headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json',} if api_key else {}

    async def text2image(self, model, prompt, height, width, num_inference_steps, seed, batch_size, refiner):
        """
        Asynchronously sends a text-to-image request to the server.

        Parameters:
            model: The model to be used for image generation.
            prompt: The prompt based on which the image is to be generated.
            height: The height of the generated image.
            width: The width of the generated image.
            num_inference_steps: The number of inference steps.
            seed: Seed for random number generation.
            batch_size: Batch size for processing.
            refiner: Whether to use a refiner or not.

        Returns:
            The JSON response from the server.
        """
        url = f"{self.base_url}/diffusion/{model}/text_to_image"
        payload = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "batch_size": batch_size,
            "refiner": refiner
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                return await response.json()

    async def image2image(self, image, model, prompt, height, width, strength, seed, batch_size):
        """
        Asynchronously sends an image-to-image request to the server.

        Parameters:
            image: The base image for the operation.
            model: The model to be used for image generation.
            prompt: The prompt based on which the image is to be modified.
            height: The height of the generated image.
            width: The width of the generated image.
            strength: The strength of the transformation.
            seed: Seed for random number generation.
            batch_size: Batch size for processing.

        Returns:
            The JSON response from the server.
        """
        url = f"{self.base_url}/diffusion/{model}/image_to_image"
        payload = {
            "image": image,
            "prompt": prompt,
            "height": height,
            "width": width,
            "strength": strength,
            "seed": seed,
            "batch_size": batch_size
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=self.headers) as response:
                return await response.json()

    async def interactive(self, model, prompt, temperature, repetition_penalty, top_p, top_k, max_tokens):
        url = f"{self.base_url}/text_generation/{model}/chat/interactive"
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens
        }
        data = json.dumps(payload)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self.headers) as response:
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            # Load the JSON from the line
                            json_data = json.loads(decoded_line)
                            yield json_data
                        except json.JSONDecodeError:
                            # If the line is not a valid JSON, ignore
                            pass


    async def completion(self, model, messages, temperature, repetition_penalty, top_p, max_tokens):
        url = f"{self.base_url}/text_generation/{model}/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        data = json.dumps(payload)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self.headers) as response:
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            # Load the JSON from the line
                            json_data = json.loads(decoded_line)
                            yield json_data
                        except json.JSONDecodeError:
                            # If the line is not a valid JSON, ignore
                            pass