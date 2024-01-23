import bittensor as bt

async def check_similarity_completion(self, uid, model, messages, completion, temperature, repetition_penalty, top_p, max_tokens):
    bt.logging.debug(f'Scoring {uid} text..')
    if len(completion) < 1:
        bt.logging.debug(f'{uid} completion = empty, score = 0')
        return False
    true_completion = ""
    async for chunk in self.sense.completion(model = model, messages = messages, temperature = temperature, repetition_penalty = repetition_penalty, top_p = top_p,max_tokens = max_tokens):
        token = chunk['text']
        true_completion+= token
    valid = False
    if true_completion == completion:
        bt.logging.debug(f'{uid} completion = validator, score = 1')
        valid = True
    else:
        bt.logging.debug(f'{uid} completion != validator, score = 0')
    return valid
