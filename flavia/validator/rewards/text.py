import bittensor as bt

async def check_similarity_completion(self, uid, model, messages, completion, temperature, repetition_penalty, top_p, max_tokens):
    bt.logging.debug(f'Scoring {uid} text..')
    true_completion = ""
    async for chunk in self.daemon.send_text_generation_completions(model = model, messages = messages, temperature = temperature, repetition_penalty = repetition_penalty, top_p = top_p,max_tokens = max_tokens):
        token = chunk['text']
        true_completion+= token
    valid = False
    if true_completion == completion:
        bt.logging.debug(f'The completion of {uid} is equal to that of the validator. Score = 1')
        valid = True
    else:
        bt.logging.debug(f'The completion of {uid} is not equal to that of the validator. Score = 0')
    return valid

async def calculate_speed_text(self, completion, duration):
    tokens = self.tokenizer.tokenize(completion)

    # Count the tokens
    number_of_tokens = len(tokens)

    # Calculate tokens per second
    tokens_per_second = number_of_tokens / duration
    return tokens_per_second