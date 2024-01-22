from src.flavia.neurons.validator.rewards.text2image import check_score_image
from src.flavia.neurons.validator.utils.text2image import generate_random_size_dimension, generate_random_step_with_bias, generate_unique_prompt, should_use_refiner
from src.flavia.neurons.validator.rewards.completion import check_similarity_completion
from src.flavia.neurons.validator.base import BaseValidator
from flavia.protocol import TextCompletion, TextToImage
import bittensor as bt
import random
import asyncio 
import torch 
from datasets import load_dataset
from src.flavia.neurons.validator.utils.completion import generate_unique_instruction, generate_max_tokens, frange, generate_random_top_p, generate_repetition_penalty
import src.flavia.neurons.validator.utils.ratelimit as ratelimit

class Text2ImageValidator(BaseValidator):
    def __init__(self, dendrite, config, subtensor, wallet: bt.wallet, sense):
        super().__init__(dendrite, config, subtensor, wallet, timeout=10, streaming=False)
        self.sense = sense
        self.query_type = "text2image"
        self.model = "dataautogpt3|OpenDalleV1.1"
        self.sbu_captions = load_dataset("sbu_captions", trust_remote_code=True)
    
    async def start_query(self, available_uids, metagraph) -> tuple[list, dict]:
        query_tasks = []
        uid_to_question = {}
        syns = {}
        parameters = {}
        for uid in available_uids:
            prompt = generate_unique_prompt(self)
            (height, width) = generate_random_size_dimension()
            random_steps = generate_random_step_with_bias()
            seed = random.randint(0, 2 ** 32 - 1)
            refiner = should_use_refiner()
            base_timeout = 8
            timeout_per_step = 1.5
            current_timeout = base_timeout + (random_steps // 10) * timeout_per_step

            parameters[uid] = {
                "model":self.model, "prompt":prompt, "seed":seed, 
                "num_inference_steps":random_steps, "height":height, 
                "width":width, "refiner":refiner
            }

            syn = TextToImage(**parameters[uid])
            syns[uid] = syn
            bt.logging.info(
                f"Sending {self.query_type} request to uid: {uid}, "
                f"with timeout {current_timeout}, "
                f"image prompt {prompt}"
            )

            task = self.query_miner(metagraph, uid, syn, timeout=current_timeout)
            query_tasks.append(asyncio.create_task(task))

        rate_limit = 0.5
        random.shuffle(query_tasks)
        query_responses = await ratelimit.run_tasks(query_tasks, rate_limit)

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
            completion = rep[1][0]
            query = parameters[uid]

            scores[uid] = 0
            uid_scores_dict[uid] = 0
            if len(completion.output) > 0:
                if await check_score_image(self, uid=uid, model=completion.model, image=completion.output[0], prompt=completion.prompt, seed=completion.seed, height=completion.height, width=completion.width, refiner=completion.refiner, steps=completion.num_inference_steps):
                    scores[uid] = 1
                    uid_scores_dict[uid] = 1
        print(scores)
        return scores, uid_scores_dict
