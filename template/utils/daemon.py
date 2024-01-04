import requests
import json
import bittensor as bt
class DaemonClient:
    def __init__(self, base_url='http://127.0.0.1:8000'):
        self.base_url = base_url

    def send_text_to_image_request(self, model, prompt, height, width, num_inference_steps, seed, batch_size, refiner):
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
        data = json.dumps(payload)
        response = requests.post(url, data=data, stream=True)
        return response.json()

    def send_image_to_image_request(self, image, model, prompt, height, width, strength, seed, batch_size):
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
        data = json.dumps(payload)
        response = requests.post(url, data=data, stream=True)
        return response.json()

    def send_text_generation_interactive(self, model, prompt, temperature, repetition_penalty, top_p, top_k, max_tokens):
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
        response = requests.post(url, data=data, stream=True)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    # Charger le JSON de la ligne
                    json_data = json.loads(decoded_line)
                    yield json_data
                except json.JSONDecodeError:
                    # Si la ligne n'est pas un JSON valide, ignorer
                    pass

    def send_text_generation_completions(self, model, messages, temperature, repetition_penalty, top_p, max_tokens):
        url = f"{self.base_url}/text_generation/{model}/chat/completions"
        payload = {
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        data = json.dumps(payload)
        response = requests.post(url, data=data, stream=True)
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    # Charger le JSON de la ligne
                    json_data = json.loads(decoded_line)
                    yield json_data
                except json.JSONDecodeError:
                    # Si la ligne n'est pas un JSON valide, ignorer
                    pass