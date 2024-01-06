import torch
import random
import bittensor as bt
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """
    Returns k or fewer available random uids from the metagraph, preferring those not already queried.
    If there are not enough unqueried uids, the function will return fewer uids.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        Resets miners_already_queried if all have been queried.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude
        uid_not_queried = uid not in self.miners_already_queried

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded and uid_not_queried:
                candidate_uids.append(uid)

    # If not enough uids, reset the queried list and re-run the function
    if len(avail_uids) == len(self.miners_already_queried):
        self.miners_already_queried.clear()

    uids = torch.tensor(random.sample(candidate_uids, min(k, len(candidate_uids))))
    self.miners_already_queried.update(uids.tolist())
    return uids