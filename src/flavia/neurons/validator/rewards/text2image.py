import bittensor as bt

from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
import base64
from PIL import Image
import io
import torchvision.transforms.functional as F
import numpy as np
def tensor_to_numpy(tensor):
    # Convert PyTorch tensor to a NumPy array
    return tensor.mul(255).byte().numpy().transpose(1, 2, 0)
transform_b64_bt = transforms.Compose([
    transforms.Lambda(lambda x: base64.b64decode(x)),
    transforms.Lambda(lambda x: Image.open(io.BytesIO(x))), 
    transforms.ToTensor() 
])
def compare_images(tensor_image1, tensor_image2):
    # Convert tensors to numpy arrays
    np_image1 = tensor_to_numpy(tensor_image1)
    np_image2 = tensor_to_numpy(tensor_image2)

    # Convert images to grayscale
    img1_gray = F.to_grayscale(F.to_pil_image(np_image1), num_output_channels=1)
    img2_gray = F.to_grayscale(F.to_pil_image(np_image2), num_output_channels=1)

    # Convert PIL images back to numpy arrays
    np_img1_gray = np.array(img1_gray)
    np_img2_gray = np.array(img2_gray)

    # Calculate the structural similarity
    similarity = ssim(np_img1_gray, np_img2_gray)

    # Check if similarity is greater than 99%
    return similarity
    
async def check_score_image(self, uid, model, image, prompt, steps, seed, height, width, refiner):
    bt.logging.debug(f'Scoring {uid} image..')
    r_output = await self.sense.text2image(model=model, prompt=prompt, height=height, width=width, num_inference_steps=steps, seed=seed, batch_size=1, refiner=refiner)
    
    vali_image = transform_b64_bt(r_output['images'][0])
    similarity_score = compare_images(bt.Tensor.deserialize(image), vali_image)

    valid = False
    if similarity_score > 0.7:
        bt.logging.debug(f'{uid} reward= 1, similarity={similarity_score}')
        valid = True
    else:
        bt.logging.debug(f'{uid} reward= 0, similarity={similarity_score}')
    return valid

async def calculate_speed_image(width, height, num_inferences_step, duration):
    # Calculate the number of pixels
    number_of_pixels = width * height
    
    # Calculate the effective duration per inference step
    step_duration = duration / num_inferences_step
    
    # Calculate the speed in pixels per unit of time with the effective duration
    speed = number_of_pixels / step_duration
    
    # Not working, to rebase
    return speed