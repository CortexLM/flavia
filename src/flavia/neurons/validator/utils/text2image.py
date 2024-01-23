import random
import numpy as np

def generate_unique_prompt(self):
    # Load the complete dataset from Hugging Face
    complete_dataset = self.sbu_captions

    # Select a random split if the dataset contains multiple splits (like train, test, etc.)
    available_splits = list(complete_dataset.keys())
    chosen_split = random.choice(available_splits)

    # Load the chosen split
    dataset_split = complete_dataset[chosen_split]

    # Choose a random index
    random_index = random.randint(0, len(dataset_split) - 1)

    # Select the element at the random index
    random_prompt = dataset_split[random_index]['caption']

    return random_prompt

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
    Generates a random inference step from the range 30 to 75 (inclusive, with steps of 5),

    :return: A randomly selected inference step.
    """
    # Creating a range from 5 to 70 (inclusive) with steps of 5
    num_inference_steps = np.arange(30, 75, 5)

    # Generating a random step
    return int(np.random.choice(num_inference_steps))