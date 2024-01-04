from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.logging import logging
from fastapi.responses import StreamingResponse

class TextInteractive(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = 1.1
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 512
    refiner: Optional[bool] = False

class TextCompletion(BaseModel):
    messages: List
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = 1.1
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    refiner: Optional[bool] = False

class TextToImage(BaseModel):
    prompt: str
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 30
    seed: Optional[int] = -1 
    batch_size: Optional[int] = 1
    refiner: Optional[bool] = False

class ImageToImage(BaseModel):
    image: str
    prompt: str
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    strength: Optional[int] = 1
    seed: Optional[int] = -1 
    batch_size: Optional[int] = 1

class DaemonAPI:
    def __init__(self, model):
        self.app = FastAPI()
        self.models = model.models

        @self.app.post("/diffusion/{model_name}/text_to_image")
        async def diffusion_text_to_image(model_name: str, interact: TextToImage):
            model = self.models.get(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            response = model.t2i(
                prompt=interact.prompt,
                height=interact.height,
                width=interact.width,
                num_inference_steps=interact.num_inference_steps,
                seed=interact.seed,
                batch_size=interact.batch_size,
                refiner=interact.refiner
            )
            return response
        
        @self.app.post("/diffusion/{model_name}/image_to_image")
        async def diffusion_image_to_image(model_name: str, interact: ImageToImage):
            model = self.models.get(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            response = model.i2i(
                image=interact.image,
                prompt=interact.prompt,
                height=interact.height,
                width=interact.width,
                strength=interact.strength,
                seed=interact.seed,
                batch_size=interact.batch_size
            )
            return response
    
        @self.app.post("/text_generation/{model_name}/chat/interactive")
        async def text_generation_interactive(model_name: str, interact: TextInteractive):
            model = self.models.get(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            response = model.interactive(
                prompt=interact.prompt,
                temperature=interact.temperature,
                repetition_penalty=interact.repetition_penalty,
                top_p=interact.top_p,
                top_k=interact.top_k,
                max_tokens=interact.max_tokens,
            )
            return StreamingResponse(response)

        @self.app.post("/text_generation/{model_name}/chat/completions")
        async def text_generation_completions(model_name: str, interact: TextCompletion):
            model = self.models.get(model_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            response = model.completion(
                messages=interact.messages,
                temperature=interact.temperature,
                repetition_penalty=interact.repetition_penalty,
                top_p=interact.top_p,
                max_tokens=interact.max_tokens,
            )
            return StreamingResponse(response)

    def run(self, host="127.0.0.1", port=8000):
        import uvicorn
        logging.success(f"Daemon started at address {host} on port {port}")
        uvicorn.run(self.app, host=host, port=port, log_level="error")