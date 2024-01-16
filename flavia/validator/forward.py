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
from flavia.protocol import TextCompletion, TextInteractive, ImageToImage, TextToImage
from flavia.utils.uids import get_random_uids
import torch
import numpy as np
import time
import random
from skimage.metrics import structural_similarity as ssim
import numpy as np
from flavia.validator.rewards.image import check_score_image, calculate_speed_image
from flavia.validator.rewards.text import check_similarity_completion, calculate_speed_text
from flavia.validator.utils.text import generate_unique_instruction, generate_max_tokens, frange, generate_random_top_p, generate_repetition_penalty
from flavia.validator.utils.image import generate_unique_prompt, generate_random_size_dimension, should_use_refiner, generate_random_step_with_bias
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
            start_time = time.time()  # Record the start time
            response = await self.dendrite(
                axons=[self.metagraph.axons[uid]],
                synapse=TextToImage(model=model, prompt=prompt, seed=seed, num_inference_steps=random_steps, height=height, width=width, refiner=refiner),
                deserialize=False,
                streaming=False,
                timeout=12
            )
            end_time = time.time()  # Record the end time
            # Calculate the duration and format it
            duration = end_time - start_time
            return uid, response, prompt, random_steps, seed, height, width, refiner, duration
        except Exception as e:
            bt.logging.error(f"Error querying miner {uid}: {e}")
            return uid, None, None, None, None, None, None, None
        
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
            for resp in response:
                i = 0
                async for chunk in resp:       
                    if isinstance(chunk, list):
                        completion += chunk[0].replace('<newline>', '\n')
                    else:
                        # last object yielded is the synapse itself with completion filled
                        synapse = chunk
                break

            end_time = time.time()  # Record the end time
            # Calculate the duration and format it
            duration = end_time - start_time
            bt.logging.debug(f"{uid} took {duration:.2f} seconds to complete")
            return uid, model, messages, completion, max_tokens, repetition_penalty, top_p, duration

        except Exception as e:
            bt.logging.error(f"Error querying miner {uid}: {e}")
            return uid, None, None, None, None, None, None, None

    # Select miner UIDs to query
    miner_uids = get_random_uids(self, k=10)

    # Run queries asynchronously
    tasks = [query_miner_image(uid) for uid in miner_uids]
    responses = await asyncio.gather(*tasks)

    # Process and score responses
    df_rewards_tensor = {}
    rewards = {}
    for uid, response, prompt, random_steps, seed, height, width, refiner, duration in responses:
        if None in [uid, response, prompt, random_steps, seed, height, width, refiner]:
            rewards[uid] = 0
            continue;
        response = response[0]
        if len(response.output) > 0:
            if await check_score_image(self, uid=uid, model=image_model, image=response.output[0], prompt=prompt, steps=random_steps, seed=seed, height=height, width=width, refiner=refiner):
                rewards[uid] = 1
                df_rewards_tensor[uid] = await calculate_speed_image(width=width, height=height, num_inferences_step=random_steps, duration=duration)
            else:
                rewards[uid] = 0
                df_rewards_tensor[uid] = 0
        else:
            rewards[uid] = 0
            df_rewards_tensor[uid] = 0

    bt.logging.info(f"Scored responses: {rewards}")
    rewards_tensor = torch.FloatTensor(list(rewards.values()))
    # rewards_tensor_df = torch.FloatTensor(list(df_rewards_tensor.values()))
    self.update_scores(rewards_tensor , miner_uids)
    # self.update_df_scores(rewards_tensor_df , miner_uids)
    # Select miner UIDs to query
    miner_uids_cp = get_random_uids(self, k=25)
    # Run queries asynchronously
    tasks_cp = [query_miner_completions(uid) for uid in miner_uids_cp]
    responses_cp = await asyncio.gather(*tasks_cp)
    rewards = {}
    cp_speed = {}
    async def process_responses(responses_cp):
        for uid, model, messages, completion, max_tokens, repetition_penalty, top_p, duration in responses_cp:
            if len(completion) > 0:
                # Assuming check_similarity_completion can be an async function
                if await check_similarity_completion(self, uid=uid, model=model, messages=messages, completion=completion, temperature=0, repetition_penalty=repetition_penalty, top_p=top_p, max_tokens=max_tokens):
                    cp_speed[uid] = await calculate_speed_text(self, completion=completion, duration=duration)
                    rewards[uid] = 1
                else:
                    cp_speed[uid] = 0
                    rewards[uid] = 0
            else:
                cp_speed[uid] = 0
                rewards[uid] = 0
        return rewards
    
    asyncio.run(process_responses(responses_cp))

    # Convertissez cp_speed en un tenseur
    cp_speed_tensor = torch.FloatTensor(list(cp_speed.values()))
    rewards_tensor = torch.FloatTensor(list(rewards.values()))

    max_cp_speed_weight = 0.3

    tolerance_rate = 0.1

    cp_speed_weight = min(max_cp_speed_weight, max_cp_speed_weight * (1 / cp_speed_tensor.max()))

    sorted_indices = cp_speed_tensor.argsort(descending=True)

    num_fastest_responses = int(len(sorted_indices) * tolerance_rate)

    if num_fastest_responses > 0:
        fastest_indices = sorted_indices[:num_fastest_responses]
        normalized_rewards = rewards_tensor.clone()
        normalized_rewards[fastest_indices] = rewards_tensor[fastest_indices] + cp_speed_weight
        normalized_rewards = (normalized_rewards - normalized_rewards.min()) / (normalized_rewards.max() - normalized_rewards.min())
    else:
        normalized_rewards = rewards_tensor.clone()

    self.update_scores(normalized_rewards, miner_uids_cp)

    bt.logging.info("rewards", normalized_rewards)