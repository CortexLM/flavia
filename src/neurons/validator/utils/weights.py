import asyncio
import itertools
import torch
import bittensor as bt
import concurrent
import traceback
import random
from typing import Awaitable

from src.neurons.validator.utils.uids import check_uid_availability
iterations_per_set_weights = 2
scoring_organic_timeout = 60

class Weights:
    def create_task(self, awaitable: Awaitable) -> asyncio.Task:
        async def _log_exception(awaitable):
            try:
                return await awaitable
            except Exception as e:
                bt.logging.error(
                    f'Encountered error:\n'
                    f'{"".join(traceback.format_exception(e))}'
                )
        return asyncio.create_task(_log_exception(awaitable))

    def __init__(self, loop: asyncio.AbstractEventLoop, dendrite, subtensor, config, wallet, dummy_validator):
        self.loop = loop
        self.dendrite = dendrite
        self.subtensor = subtensor
        self.config = config
        self.wallet = wallet
        self.dummy_validator = dummy_validator

        self.moving_average_scores = None
        self.metagraph = subtensor.metagraph(config.netuid)
        self.total_scores = torch.zeros(len(self.metagraph.hotkeys))
        self.organic_scoring_tasks = set()

        self.thread_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='asyncio')
        self.create_task(self.consume_organic_scoring())
        self.create_task(self.perform_synthetic_scoring_and_update_weights())

    async def consume_organic_scoring(self):
        while True:
            try:
                if self.organic_scoring_tasks:
                    completed, _ = await asyncio.wait(self.organic_scoring_tasks, timeout=1,
                                                      return_when=asyncio.FIRST_COMPLETED)
                    for task in completed:
                        if task.exception():
                            bt.logging.error(
                                f'Encountered error:\n'
                                f'{"".join(traceback.format_exception(task.exception()))}'
                            )
                        else:
                            success, data = task.result()
                            if not success:
                                continue
                            self.total_scores += data[0]
                    self.organic_scoring_tasks.difference_update(completed)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                bt.logging.error(f'Encountered in {self.consume_organic_scoring.__name__} loop:\n{traceback.format_exc()}')
                await asyncio.sleep(10)
    
    def select_validator(self):
        return self.dummy_validator
    
    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        tasks = {uid.item(): check_uid_availability(self.metagraph, self.metagraph.axons[uid.item()], uid.item(), vpermit_tao_limit=5000) for uid in self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())
        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

        return available_uids

    async def update_weights(self, steps_passed):
        """ Update weights based on total scores, using min-max normalization for display. """
        bt.logging.info("Updating weights")

        # Calculate average scores
        avg_scores = self.total_scores / (steps_passed + 1)

        # Normalize avg_scores to a range of 0 to 1
        min_score = torch.min(avg_scores)
        max_score = torch.max(avg_scores)

        # Check for division by zero and normalize scores
        if max_score > min_score:
            normalized_scores = (avg_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = torch.zeros_like(avg_scores)

        bt.logging.info(f"Normalized scores: {normalized_scores}")

        # Update weights using average scores
        await self.set_weights(avg_scores)

    async def set_weights(self, scores):
        """ Update the moving average of weights based on new scores. """
        alpha = 0.3

        if self.moving_average_scores is None:
            self.moving_average_scores = scores.clone()

        # Check if there is a size mismatch
        if scores.size(0) != self.moving_average_scores.size(0):
            bt.logging.warning(f"Size mismatch: scores size {scores.size()}, moving_average_scores size {self.moving_average_scores.size()}")
            # Determine which tensor is smaller
            if scores.size(0) > self.moving_average_scores.size(0):
                
                # Pad self.moving_average_scores with zeros
                padding_size = scores.size(0) - self.moving_average_scores.size(0)
                padding = torch.zeros(padding_size, dtype=self.moving_average_scores.dtype)
                self.moving_average_scores = torch.cat([self.moving_average_scores, padding])
            else:
                # Pad scores with zeros
                padding_size = self.moving_average_scores.size(0) - scores.size(0)
                padding = torch.zeros(padding_size, dtype=scores.dtype)
                scores = torch.cat([scores, padding])
        self.moving_average_scores = alpha * scores + (1 - alpha) * self.moving_average_scores
        bt.logging.info(f"Updated moving average of weights: {self.moving_average_scores}")

        # Create a wrapper function for self.subtensor.set_weights
        def set_weights_sync():
            return self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=self.moving_average_scores,
                wait_for_inclusion=False,
            )

        # Use run_in_executor to execute the synchronous function in another thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, set_weights_sync)
        bt.logging.success("Successfully set weights.")

    async def perform_synthetic_scoring_and_update_weights(self):
        while True:
            for steps_passed in itertools.count():
                def run_metagraph_sync():
                    return self.subtensor.metagraph(
                        self.config.netuid
                    )
                loop = asyncio.get_event_loop()
                self.metagraph = await loop.run_in_executor(None, run_metagraph_sync)
                available_uids = await self.get_available_uids()
                selected_validator = self.select_validator()
                scores, _ = await self.process_modality(selected_validator, available_uids)
                if scores.size(0) > self.total_scores.size(0):
                    size_diff = scores.size(0) - self.total_scores.size(0)
                    self.total_scores = torch.cat([self.total_scores, torch.zeros(size_diff)], dim=0)

                self.total_scores += scores

                steps_since_last_update = steps_passed % iterations_per_set_weights

                if steps_since_last_update == iterations_per_set_weights - 1:
                    await self.update_weights(steps_passed)
                else:
                    bt.logging.info(
                        f"Updating weights in {iterations_per_set_weights - steps_since_last_update - 1} iterations."
                    )

                await asyncio.sleep(10)

    def shuffled(self, list_: list) -> list:
        list_ = list_.copy()
        random.shuffle(list_)
        return list_
    
    async def process_modality(self, selected_validator, available_uids):
        uid_list = self.shuffled(list(available_uids.keys()))
        bt.logging.info(f"starting {selected_validator.__class__.__name__} get_and_score for {uid_list}")
        scores, uid_scores_dict = await selected_validator.get_and_score(uid_list, self.metagraph)
        return scores, uid_scores_dict

