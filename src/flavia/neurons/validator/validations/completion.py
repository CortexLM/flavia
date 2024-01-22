from src.flavia.neurons.validator.rewards.completion import check_similarity_completion
from src.flavia.neurons.validator.base import BaseValidator
from flavia.protocol import TextCompletion
import bittensor as bt
import random
import asyncio 
import torch 
from datasets import load_dataset
from src.flavia.neurons.validator.utils.completion import generate_unique_instruction, generate_max_tokens, frange, generate_random_top_p, generate_repetition_penalty

class TextCompletionValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet, sense):
        super().__init__(dendrite, config, subtensor, wallet, timeout=10, streaming=True)
        self.sense = sense
        self.query_type = "text"
        self.model = "Qwen|Qwen-72B-Chat"
        self.instructions_dataset = iter(load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", trust_remote_code=True)['train'].shuffle(seed=random.randint(0, 1000000)).to_iterable_dataset())

    async def handle_response(self, uid: str, responses) -> tuple[str, str]:
        completion = ""
        for resp in responses:
            try:
                async for chunk in resp:
                    if isinstance(chunk, list):
                        completion += chunk[0].replace('<newline>', '\n')
            except asyncio.CancelledError:
                # Handle the CancelledError as needed, e.g., set completion to ""
                completion = ""
                bt.logging.error(f"Error during {uid} completion")
            break

        return uid, completion, responses
    
    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}
        syns = {}
        parameters = {}
        for uid in available_uids:
            max_tokens = generate_max_tokens();
            top_p = generate_random_top_p()
            instruction, input = generate_unique_instruction(self)
            repetition_penalty = generate_repetition_penalty()
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input}
            ]
            parameters[uid] = {
                "model": self.model, "messages":messages, "temperature":0, "top_p":top_p, "max_tokens":max_tokens, "repetition_penalty": repetition_penalty
            }
            syn = TextCompletion(**parameters[uid])
            syns[uid] = syn
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid}, "
                f"with timeout {self.timeout}\n"
                f"completion {syn.messages}"
            )
            task = self.query_miner(metagraph, uid, syn)
            query_tasks.append(task)
        query_responses = await asyncio.gather(*query_tasks)
        return query_responses, uid_to_question, parameters    
    
    async def score_responses(
        self,
        query_responses: list[tuple[int, str]],  # [(uid, response)]
        uid_to_question: dict[int, str],  # uid -> prompt
        metagraph: bt.metagraph,
        parameters: dict,
    ) -> tuple[torch.Tensor, dict[int, float], dict]:
        scores = torch.zeros(len(metagraph.hotkeys))
        uid_scores_dict = {}
        response_tasks = []
        scoring_tasks = []
        for rep in query_responses:
            uid = rep[0]
            completion = rep[1]
            query = parameters[uid]
            if await check_similarity_completion(self, uid=uid, model=query['model'], messages=query['messages'], completion=completion, temperature=0, repetition_penalty=query['repetition_penalty'], top_p=query['top_p'], max_tokens=query['max_tokens']):
                scores[uid] = 1
                uid_scores_dict[uid] = 1
            else:
                scores[uid] = 0
                uid_scores_dict[uid] = 0
        print(scores)
        return scores, uid_scores_dict
