import aiohttp
import json
import asyncio
class DaemonClient:
    def __init__(self, base_url='http://127.0.0.1:8000', api_key=None):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json',} if api_key else {}

    async def send_request_with_retry(self, url, payload, max_retries=1, timeout=15):
        try_count = 0
        while try_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=self.headers, timeout=timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise aiohttp.HttpProcessingError()
            except (aiohttp.ClientError, aiohttp.HttpProcessingError, json.JSONDecodeError):
                try_count += 1
                await asyncio.sleep(1)

        return None
    
    async def send_stream_request_with_retry(self, url, payload, max_retries=2, timeout=15):
        try_count = 0
        while try_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=self.headers, timeout=timeout) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line:
                                    decoded_line = line.decode('utf-8')
                                    try:
                                        json_data = json.loads(decoded_line)
                                        yield json_data
                                    except json.JSONDecodeError:
                                        pass  # Gérer l'erreur de décodage si nécessaire
                        else:
                            raise aiohttp.HttpProcessingError()
            except (aiohttp.ClientError, aiohttp.HttpProcessingError):
                try_count += 1
                await asyncio.sleep(1)    
    async def send_text_to_image_request(self, model, prompt, height, width, num_inference_steps, seed, batch_size, refiner):
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
        return await self.send_request_with_retry(url, payload)

    async def send_image_to_image_request(self, image, model, prompt, height, width, strength, seed, batch_size):
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
        return await self.send_request_with_retry(url, payload)

    async def send_text_generation_interactive(self, model, prompt, temperature, repetition_penalty, top_p, top_k, max_tokens):
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
        
        return self.send_stream_request_with_retry(url, payload)


    async def send_text_generation_completions(self, model, messages, temperature, repetition_penalty, top_p, max_tokens):
        url = f"{self.base_url}/text_generation/{model}/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        data = json.dumps(payload)
        return self.send_stream_request_with_retry(url, payload)