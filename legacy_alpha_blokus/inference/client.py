from typing import Tuple
import time
import random
import logging
import numpy as np
import asyncio

from alpha_blokus.inference.actor import InferenceActor

class InferenceClient:
    def __init__(self, actor: InferenceActor, batch_size: int, cfg: dict):
        self.batch_size = batch_size
        self.actor = actor
        self.cfg = cfg
        
        # This fills up as evaluation requests come in. When full, we submit to
        # an external process. We don't send these one-by-one because each remote()
        # request has some overhead.
        self.evaluation_batch = []
        self.move_indices_batch = []
        self.evaluation_batch_ids = []
        self.futures = {}

        # These fields are set by init_in_process.
        self.loop = None

    async def evaluate(self, board: np.ndarray, move_indices: np.ndarray, turn: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        board - An occupancies board rolled and rotated into the perspective of the current player.
        move_indices - The move indices (as could be played on the rotated/rolled board) that we're interested
                       in the values for. Usually, this is the list of the valid moves on the rotated/rolled board.
                       This parameter is necessary for caching purposes, so that we only cache the moves that are
                       relevant. This method returns the policy logits in the same order as this list.
        """
        evaluation_id = random.getrandbits(60)
        future = self.loop.create_future()
        self.futures[evaluation_id] = future

        self.evaluation_batch.append(board)
        self.move_indices_batch.append(move_indices)
        self.evaluation_batch_ids.append(evaluation_id)

        # Check if we're ready to evaluate the batch based on the batch size and the
        # number of evaluations currently waiting on cached values.
        if len(self.evaluation_batch) >= self.batch_size:
            evaluation_params = self._fetch_and_clear_evaluation_params()
            await self._evaluate_batch(*evaluation_params)

        result = await future
        if evaluation_id:
            del self.futures[evaluation_id]

        return result

    def init_in_process(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    def _fetch_and_clear_evaluation_params(self):
        # Make local copies of all the relevant values, and then reset 
        # the self.* copies. This way, when we switch to a different async
        # thread upon calling the await below, we don't need to worry about 
        # our values here getting clobbered.
        #
        # I believe the [:] isn't necessary below, but I'm including it just
        # to be safe.
        # 
        # This function is not async, to ensure it runs before the event loop might
        # interrupt.
        evaluation_batch_np = np.stack(self.evaluation_batch)
        evaluation_batch_ids = self.evaluation_batch_ids[:]
        move_indices_batch = self.move_indices_batch[:]

        self.evaluation_batch = []
        self.move_indices_batch = []
        self.evaluation_batch_ids = []

        return evaluation_batch_np, evaluation_batch_ids, move_indices_batch

    async def _evaluate_batch(self, evaluation_batch_np, evaluation_batch_ids, move_indices_batch):
        values, policy_logits = await self.actor.evaluate_batch.remote(evaluation_batch_np)

        assert len(values) == len(evaluation_batch_ids)

        for value, policy_logit, move_indices, evaluation_id in zip(
            values,
            policy_logits,
            move_indices_batch,
            evaluation_batch_ids,
        ):
            filtered_policy_logits = policy_logit[move_indices]
            future = self.futures[evaluation_id]
            future.set_result((value, filtered_policy_logits))
