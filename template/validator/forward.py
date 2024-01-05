# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Copyright © 2023 Cortex Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import asyncio
from template.protocol import TextCompletion, TextInteractive, ImageToImage, TextToImage
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import time
import random
import torchvision.transforms.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import io
import base64
def tensor_to_numpy(tensor):
    # Convert PyTorch tensor to a NumPy array
    return tensor.mul(255).byte().numpy().transpose(1, 2, 0)

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
def generate_unique_prompt(self):
    # from ImageSubnet
    initial_prompt = next(self.diffusiondb)['prompt']
    initial_prompt = initial_prompt.split(' ')
    keep = random.randint(1, len(initial_prompt))
    keep = min(keep, 8)
    initial_prompt = ' '.join(initial_prompt[:keep])
    prompt = (self.magic_prompt( initial_prompt, pad_token_id = 50256)[0]['generated_text']).replace('\n', '')
    return prompt

def generate_unique_instruction(self):
    # from ImageSubnet
    dataset = next(self.instructions_dataset)
    instruction = dataset['instruction']
    input = dataset['input']
    return instruction, input

def generate_random_size_dimension():
    # List of aspect ratios and corresponding sizes
    sizes = [
        (1024, 1024),  # 1:1 Square
        (1152, 896),   # 9:7
        (896, 1152),   # 7:9
        (832, 1216),   # 13:19
        (512, 512), # 1:1 Square
        (768, 768) # 1:1 Square 
    ]

    # Select a random size from the list
    return random.choice(sizes)

def should_use_refiner():
    """
    Function to determine with a 25% chance whether to use a refiner or not.

    Returns:
        bool: True if a refiner should be used, False otherwise.
    """
    return random.random() < 0.25

transform_b64_bt = transforms.Compose([
    transforms.Lambda(lambda x: base64.b64decode(x)),
    transforms.Lambda(lambda x: Image.open(io.BytesIO(x))), 
    transforms.ToTensor() 
])
def generate_random_step_with_bias():
    """
    Generates a random inference step from the range 5 to 70 (inclusive, with steps of 5),
    with numbers less than 35 being 4 times more likely.

    :return: A randomly selected inference step.
    """
    # Creating a range from 5 to 70 (inclusive) with steps of 5
    num_inference_steps = np.arange(5, 75, 5)

    # Splitting the range into two parts: less than 35 and 35 or more
    less_than_35 = num_inference_steps[num_inference_steps < 25]
    thirty_five_or_more = num_inference_steps[num_inference_steps >= 25]

    # Adjusting probabilities: 4 times more likely for the 'less_than_35' group
    prob_less_than_35 = 4 / (4 + 1)
    prob_thirty_five_or_more = 1 / (4 + 1)

    # Assigning probabilities to each group
    probabilities = np.array([prob_less_than_35 / len(less_than_35)] * len(less_than_35) + 
                             [prob_thirty_five_or_more / len(thirty_five_or_more)] * len(thirty_five_or_more))

    # Generating a random step
    return int(np.random.choice(num_inference_steps, p=probabilities))

async def check_similarity_completion(self, uid, model, messages, completion, temperature, repetition_penalty, top_p, max_tokens):
    bt.logging.debug(f'Scoring {uid} text..')
    true_completion = ""
    for chunk in self.daemon.send_text_generation_completions(model = model, messages = messages, temperature = temperature, repetition_penalty = repetition_penalty, top_p = top_p,max_tokens = max_tokens):
        token = chunk['text']
        true_completion+= token
    valid = False
    if true_completion == completion:
        bt.logging.debug(f'The completion of {uid} is equal to that of the validator. Score = 1')
        valid = True
    else:
        bt.logging.debug(f'The completion of {uid} is not equal to that of the validator. Score = 0')
    return valid
def check_score_image(self, uid, model, image, prompt, steps, seed, height, width, refiner):
    bt.logging.debug(f'Scoring {uid} image..')
    r_output = self.daemon.send_text_to_image_request(model=model, prompt=prompt, height=height, width=width, num_inference_steps=steps, seed=seed, batch_size=1, refiner=refiner)
    
    vali_image = transform_b64_bt(r_output['images'][0])
    similarity_score = compare_images(bt.Tensor.deserialize(image), vali_image)

    valid = False
    if similarity_score > 0.7:
        bt.logging.debug(f'The image of {uid} is equal to that of the validator. Score = 1')
        valid = True
    else:
        bt.logging.debug(f'The image of {uid} is not equal to that of the validator. Score = 0')
    return valid
def generate_max_tokens():
    # Define the range for max_tokens
    min_tokens = 128
    max_tokens = 712

    # Generating a list of multiples of 8 within the specified range
    valid_tokens = [i for i in range(min_tokens, max_tokens + 1) if i % 8 == 0]

    # Randomly selecting a value from the li+st of valid max_tokens
    return random.choice(valid_tokens)

def frange(start, stop, step):
    while start < stop:
        yield round(start, 1)
        start += step
def generate_random_top_p():
    # Define the range and step for repetition penalty
    min_top_p = 0.1
    max_top_p = 0.9
    step = 0.1

    # Generating a list of valid penalties within the range with the specified step
    valid_top_p = [round(i, 1) for i in frange(min_top_p, max_top_p + step, step)]

    # Randomly selecting a value from the list of valid penalties
    return random.choice(valid_top_p)

def generate_repetition_penalty():
    # Define the range and step for repetition penalty
    min_penalty = 1.0
    max_penalty = 1.2
    step = 0.1

    # Generating a list of valid penalties within the range with the specified step
    valid_penalties = [round(i, 1) for i in frange(min_penalty, max_penalty + step, step)]

    # Randomly selecting a value from the list of valid penalties
    return random.choice(valid_penalties)
async def forward(self):
    
    """
    The forward function is called by the validator every time step.
    It queries the network asynchronously and scores the responses.
    """
    image_model = "dataautogpt3|OpenDalleV1.1"
    text_model = "Qwen|Qwen-72B-Chat"
    # Asynchronously query a set of miner UIDs
    async def query_miner_image(uid):
        try:
            # Generate a unique prompt for each miner
            model = image_model
            
            prompt = generate_unique_prompt(self)
            (height, width) = generate_random_size_dimension()
            random_steps = generate_random_step_with_bias()
            seed = random.randint(0, 2 ** 32 - 1)
            refiner = should_use_refiner()
            # Perform the query
            response = await self.dendrite(
                axons=[self.metagraph.axons[uid]],
                synapse=TextToImage(model=model, prompt=prompt, seed=seed, num_inference_steps=random_steps, height=height, width=width, refiner=refiner),
                deserialize=False,
                streaming=False,
                timeout=12
            )
            return uid, response, prompt, random_steps, seed, height, width, refiner
        except Exception as e:
            bt.logging.error(f"Error querying miner {uid}: {e}")
            return uid, None
        
    async def query_miner_completions(uid):

        try:
            # Generate a unique prompt for each miner
            model = text_model
            max_tokens = generate_max_tokens();
            top_p = generate_random_top_p()
            instruction, input = generate_unique_instruction(self)
            repetition_penalty = generate_repetition_penalty()
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input}
            ]

            # Perform the query
            response = await self.dendrite(
                axons=[self.metagraph.axons[uid]],
                synapse=TextCompletion(model=model, messages=messages, temperature=0, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty),
                deserialize=False,
                streaming=True,
                timeout=15
            )
            start_time = time.time()  # Record the start time

            completion = ""
            try:
            for resp in response:
                i = 0
                async for chunk in resp:       
                    if isinstance(chunk, list):
                        completion += chunk[0].replace('<newline>', '\n')
                    else:
                        # last object yielded is the synapse itself with completion filled
                        synapse = chunk
                break
            except asyncio.CancelledError:
                # En cas d'annulation, fixer les valeurs spécifiques et sortir
                duration = 15
                completion = ""
                return uid, model, messages, completion, max_tokens, repetition_penalty, top_p, duration

            end_time = time.time()  # Record the end time
            # Calculate the duration and format it
            duration = end_time - start_time
            bt.logging.debug(f"{uid} took {duration:.2f} seconds to complete")
            return uid, model, messages, completion, max_tokens, repetition_penalty, top_p, duration

        except Exception as e:
            bt.logging.error(f"Error querying miner {uid}: {e}")
            return uid, None

    # Select miner UIDs to query
    miner_uids = get_random_uids(self, k=15)

    # Run queries asynchronously
    tasks = [query_miner_image(uid) for uid in miner_uids]
    responses = await asyncio.gather(*tasks)

    # Process and score responses
    rewards = {}
    for uid, response, prompt, random_steps, seed, height, width, refiner in responses:
        response = response[0]
        if len(response.output) > 0:
            if check_score_image(self, uid=uid, model=image_model, image=response.output[0], prompt=prompt, steps=random_steps, seed=seed, height=height, width=width, refiner=refiner):
                rewards[uid] = 1
            else:
                rewards[uid] = 0
        else:
            rewards[uid] = 0

    bt.logging.info(f"Scored responses: {rewards}")
    rewards_tensor = torch.FloatTensor(list(rewards.values()))
    self.update_scores(rewards_tensor , miner_uids)

    # Select miner UIDs to query
    miner_uids = get_random_uids(self, k=15)

    # Run queries asynchronously
    tasks_cp = [query_miner_completions(uid) for uid in miner_uids]
    responses_cp = await asyncio.gather(*tasks_cp)
    rewards = {}
    async def process_responses(responses_cp):
        for uid, model, messages, completion, max_tokens, repetition_penalty, top_p, duration in responses_cp:
            if len(completion) > 0:
                # Assuming check_similarity_completion can be an async function
                if await check_similarity_completion(self, uid=uid, model=model, messages=messages, completion=completion, temperature=0, repetition_penalty=repetition_penalty, top_p=top_p, max_tokens=max_tokens):
                    rewards[uid] = 1
                else:
                    rewards[uid] = 0
            else:
                rewards[uid] = 0
        return rewards
    
    asyncio.run(process_responses(responses_cp))
    rewards_tensor = torch.FloatTensor(list(rewards.values()))
    self.update_scores(rewards_tensor , miner_uids)

    bt.logging.info("rewards", rewards_tensor)
