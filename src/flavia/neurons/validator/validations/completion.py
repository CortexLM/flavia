from src.flavia.neurons.validator.base import BaseValidator
from flavia.protocol import TextCompletion
import bittensor as bt
import random
import asyncio 
import torch 
from datasets import load_dataset
from src.flavia.neurons.validator.utils.completion import generate_unique_instruction, generate_max_tokens, frange, generate_random_top_p, generate_repetition_penalty

class TextCompletionValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=10, streaming=False)
        self.query_type = "text"
        self.model = "Qwen|Qwen-72B-Chat"
        self.instructions_dataset = iter(load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", trust_remote_code=True)['train'].shuffle(seed=random.randint(0, 1000000)).to_iterable_dataset())

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}

        for uid in available_uids:
            max_tokens = generate_max_tokens();
            top_p = generate_random_top_p()
            instruction, input = generate_unique_instruction(self)
            repetition_penalty = generate_repetition_penalty()
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input}
            ]
            syn = TextCompletion(model=self.model, messages=messages, temperature=0, top_p=top_p, max_tokens=max_tokens, repetition_penalty=repetition_penalty)
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}"
                f"completion {self.messages}"
            )
            task = self.query_miner(metagraph, uid, syn)
            query_tasks.append(task)
        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question        
    
    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
    ) -> tuple[torch.Tensor, dict[int, float], dict]:
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        response_tasks = []
        scoring_tasks = []

        for uu in query_responses:
            uid = uu[0]
            synapse = uu[1]
            if uid == 0:
                scores[uid] = 1
                uid_scores_dict[uid] = 1
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
        print(scores)
        return scores, uid_scores_dict
