import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import bittensor as bt
import torch


class BaseValidator(ABC):
    def __init__(self, dendrite: bt.dendrite, config: bt.config, subtensor: bt.subtensor, wallet: bt.wallet, timeout: int, streaming: bool):
        self.dendrite = dendrite
        self.config = config
        self.subtensor = subtensor
        self.wallet = wallet
        self.timeout = timeout
        self.streaming = streaming
        self.async_lock = asyncio.Lock()
        self.threading_lock = threading.Lock()

    async def query_miner(self, metagraph, uid, syn, timeout=None):
        try:
            current_timeout = self.timeout
            if timeout != None:
                current_timeout = timeout
            responses = await self.dendrite([metagraph.axons[uid]], syn, deserialize=False, timeout=current_timeout,
                                            streaming=self.streaming)
            return await self.handle_response(uid, responses)

        except Exception as e:
            bt.logging.error(f"Exception during query for uid {uid}: {e}")
            return uid, None
    
    async def handle_response(self, uid, responses):
        return uid, responses

    async def get_and_score(self, available_uids, metagraph):
        bt.logging.info("starting query")
        query_responses, uid_to_question, parameters = await self.start_query(available_uids, metagraph)
        bt.logging.info("scoring query")
        return await self.score_responses(query_responses, uid_to_question, metagraph, parameters)

    @abstractmethod
    async def score_responses(self, responses):
        ...