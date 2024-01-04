import random
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler
from sfast.compilers.diffusion_pipeline_compiler import compile, CompilationConfig
from diffusers.utils import load_image
from loguru import logger as logging
from fastapi import FastAPI
import inspect
import uvicorn
import argparse
import io
import time
import base64
from PIL import Image
from pydantic import BaseModel
from io import BytesIO

# Function to convert base64 to a PIL Image object
def base64_to_image(base64_encoded_image):
    image_data = base64.b64decode(base64_encoded_image)
    return Image.open(BytesIO(image_data))

# Function to convert a PIL Image object to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Check if the script is running under Uvicorn
def is_running_under_uvicorn():
    return any('uvicorn' in frame.filename for frame in inspect.stack())

# Singleton class for SDFastAPI
class SDFastAPI:
    def __init__(self, model_name, pipeline, warm_up=True):
        self.model_name = model_name
        self.pipeline = pipeline
        self.model = self.load_model(AutoPipelineForText2Image if pipeline == "t2i" else AutoPipelineForImage2Image)
        self.config = CompilationConfig.Default()
        self.configure_performance()
        self.model = compile(self.model, self.config)
        logging.warning('First diffusion generation will generate warnings. This does not compromise daemon operation.')
        self.warm_up() if warm_up else None
        logging.info(f'{self.model_name} loaded.')

    def load_model(self, pipeline):
        logging.info(f'Loading {self.model_name} diffusion model..')
        model = pipeline.from_pretrained(self.model_name, torch_dtype=torch.float16)
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
        model.safety_checker = None  # Disable the safety checker
        model.to("cuda")
        return model

    def configure_performance(self):
        self.config.enable_xformers = self.try_import_module('xformers')
        self.config.enable_triton = self.try_import_module('triton')
        self.config.enable_cuda_graph = True

    def try_import_module(self, module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            logging.error(f'{module_name} not installed, skipping')
            return False

    def warm_up(self):
        logging.info(f"Warming up {self.model_name}.. Please wait.")
        _ = self.model(**self.get_default_parameters()).images[0]

    def get_default_parameters(self):
        url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
        init_image = load_image(url).convert("RGB")
        return dict(
            image=init_image,
            prompt='Petals',
            height=1024,
            width=1024,
            num_inference_steps=50,
            num_images_per_prompt=1,
            batch_size=1,
            strength=1
        )

    def inference(self, **kwargs):
        if kwargs.get('seed', -1) == -1:
            kwargs['seed'] = random.randint(0, 2 ** 32 - 1)
        kwargs['generator'] = torch.Generator("cuda").manual_seed(kwargs['seed'])
        output_images = self.model(**kwargs).images
        return output_images

# FastAPI initialization
api = FastAPI()

# Argument parser for configurations
parser = argparse.ArgumentParser(description='Configure the SD Fast API service')
parser.add_argument('--model_name', type=str, default='segmind/Segmind-Vega')
parser.add_argument('--model_refiner', type=str, default='segmind/Segmind-Vega')
parser.add_argument('--model_type', type=str, default='t2i')
parser.add_argument('--host', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6000)
args = parser.parse_args()

# Global variable to store the SDFastAPI instance
sd_fast_api = SDFastAPI(model_name=args.model_name, pipeline=args.model_type) if is_running_under_uvicorn() else None
sd_fast_api_refiner = SDFastAPI(model_name=args.model_refiner, pipeline="i2i") if is_running_under_uvicorn() and args.model_refiner else None

# Pydantic models for API requests
class TextToImage(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    seed: int = -1 
    batch_size: int = 1
    refiner: bool = False

class ImageToImage(BaseModel):
    image: str  # base64
    prompt: str
    height: int = 1024
    width: int = 1024
    strength: int = 1
    seed: int = -1 
    batch_size: int = 1

# API endpoints
@api.get("/ping")
def ping():
    return {"message": "Hello world!"}

@api.post("/text_to_image")
def text_to_image(request: TextToImage):
    start_time = time.time()

    logging.debug(f"[-->] (Text2Image) [{sd_fast_api.model_name}] Request for Image Generation")
    # Implement logic to handle large GPU requirements if necessary
    if sd_fast_api.pipeline != "t2i":
        return {"error": "Text To Image is not supported for this model"}
    if request.width + request.height > 2048 or request.batch_size > 1:
        logging.error('GPU requirements too high')
        return {"error": "Image dimensions or batch size too large for GPU"}

    output_images = sd_fast_api.inference(prompt=request.prompt,
                                          height=request.height,
                                          width=request.width,
                                          num_inference_steps=request.num_inference_steps,
                                          seed=request.seed,
                                          batch_size=request.batch_size)

    base64_images = []
    for img in output_images:
        if request.refiner and sd_fast_api_refiner:
            logging.debug(f"[<-->] (Refiner) Applying refiner")
            img = sd_fast_api_refiner.inference(image=img,
                                                prompt=request.prompt,
                                                height=request.height,
                                                width=request.width,
                                                seed=request.seed,
                                                batch_size=request.batch_size)[0]

        base64_str = image_to_base64(img)
        base64_images.append(base64_str)

    end_time = time.time()
    processing_time = end_time - start_time
    logging.debug(f"[<--] (Text2Image) Processing Time: {round(processing_time, 2)} seconds")

    return {"images": base64_images, "processing_time": processing_time}


@api.post("/image_to_image")
def image_to_image(request: ImageToImage):
    start_time = time.time()
    logging.debug(f"[-->] (Image2Image) [{sd_fast_api.model_name}] Request for Image Generation")
    pipeline = sd_fast_api
    if sd_fast_api.pipeline == "t2i":
        pipeline = sd_fast_api_refiner
    output_images = pipeline.inference(image=base64_to_image(request.image),
                                          prompt=request.prompt, 
                                          height=request.height, 
                                          width=request.width, 
                                          strength=request.strength,
                                          seed=request.seed, 
                                          batch_size=request.batch_size)
    base64_images = [image_to_base64(img) for img in output_images]
    end_time = time.time()
    processing_time = end_time - start_time
    logging.debug(f"[<--] (Image2Image) Processing Time: {round(processing_time, 2)} seconds")
    return {"images": base64_images}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("sdfast:api", host=args.host, port=args.port, log_level="error")