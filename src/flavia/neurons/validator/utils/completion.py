import random

def generate_unique_instruction(self):
    # Load the complete dataset from Hugging Face
    complete_dataset = self.instructions_dataset

    # Select a random split if the dataset contains multiple splits (like train, test, etc.)
    available_splits = list(complete_dataset.keys())
    chosen_split = random.choice(available_splits)

    # Load the chosen split
    dataset_split = complete_dataset[chosen_split]

    # Choose a random index
    random_index = random.randint(0, len(dataset_split) - 1)

    # Select the element at the random index
    random_element = dataset_split[random_index]
    instruction = random_element['instruction']
    input = random_element['input']
    return instruction, input

def generate_max_tokens():
    # Define the range for max_tokens
    min_tokens = 128
    max_tokens = 448

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