import random
import numpy as np

def generate_unique_prompt(self):
    # from ImageSubnet
    initial_prompt = next(self.diffusiondb)['prompt']
    initial_prompt = initial_prompt.split(' ')
    keep = random.randint(1, len(initial_prompt))
    keep = min(keep, 8)
    initial_prompt = ' '.join(initial_prompt[:keep])
    prompt = (self.magic_prompt( initial_prompt, pad_token_id = 50256)[0]['generated_text']).replace('\n', '')
    return prompt

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
    # return random.random() < 0.25
    return True

def generate_random_step_with_bias():
    """
    Generates a random inference step from the range 5 to 70 (inclusive, with steps of 5),
    with numbers less than 35 being 4 times more likely.

    :return: A randomly selected inference step.
    """
    # Creating a range from 5 to 70 (inclusive) with steps of 5
    num_inference_steps = np.arange(30, 75, 5)

    # Generating a random step
    return int(np.random.choice(num_inference_steps))