# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Copyright © 2023 Cortex Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time

# Bittensor
import bittensor as bt

# Bittensor Validator Template:
import flavia
from flavia.validator import forward
import random
# import base validator class which takes care of most of the boilerplate
from flavia.base.validator import BaseValidatorNeuron
from flavia.utils.daemon import DaemonClient
# For prompt generation
from transformers import pipeline, AutoTokenizer
# Load prompt dataset.
from datasets import load_dataset
class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.diffusiondb = iter(load_dataset("poloclub/diffusiondb", trust_remote_code=True)['train'].shuffle(seed=random.randint(0, 1000000)).to_iterable_dataset())
        self.instructions_dataset = iter(load_dataset("cognitivecomputations/dolphin", "flan1m-alpaca-uncensored", trust_remote_code=True)['train'].shuffle(seed=random.randint(0, 1000000)).to_iterable_dataset())
        self.magic_prompt = pipeline("text-generation", model="Gustavosta/MagicPrompt-Dalle")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-72B-Chat", trust_remote_code=True)
        self.update_average = False
        self.daemon = DaemonClient(base_url=self.config.sense.base_url, api_key=self.config.sense.api_key)
        self.miners_already_queried = set()


    async def forward(self):
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(60*60*60*365*100)
