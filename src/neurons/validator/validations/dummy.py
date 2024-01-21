from src.neurons.validator.base import BaseValidator
from template.protocol import Dummy
import bittensor as bt
import random
import asyncio 
import torch 

class DummyValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet):
        super().__init__(dendrite, config, subtensor, wallet, timeout=10, streaming=False)
        self.query_type = "text"

    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}

        for uid in available_uids:
            syn = Dummy(dummy_input=1)
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid}, "
                f"timeout {self.timeout}: {syn.dummy_input}"
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
