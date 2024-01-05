<div align="center">

# **🌸 Flavia - Model Inference Subnet**
Execute large-scale models on the Bittensor network

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

#
### 🔴 This is an alpha version. Please report any bugs or suggest any improvements.
### 📋 Introduction
Introduction: **Bittensor Subnet 17** Flavia, A Decentralized Model Inference Subnet.

The Flavia project is a sub-network of the Bittensor network, focused on decentralized model inference at different scales.
## 🔧 Installation

**Clone the Repository:** Begin by cloning the Flavia repository to your local machine. Open your terminal and run the following command:

```git clone https://github.com/CortexLM/flavia.git```

**Navigate to the Project Directory:** Change your current working directory to the Flavia project:

```cd flavia```

**Install the Project:** Use pip to install the project. Run:

```pip install -e .```

**Run a miner:**

```pm2 start neurons/miner.py --netuid 17 --wallet.name <coldkey> --wallet.hotkey <hotkey> --sense.base_url http://host:port --sense.api_key (your api key) --logging.debug```

**Run a validator:**

```pm2 start neurons/validator.py --netuid 17 --wallet.name <coldkey> --wallet.hotkey <hotkey> --sense.base_url http://host:port --sense.api_key (your api key) --logging.debug```

## 🔧 Daemon Installation (Inference)

**Please check https://github.com/CortexLM/sense**

## 📋 Requirements

TODO

## ⚖️ Support our Subnet: Assigning weights
As a validator in the Bittensor network, your support is crucial for the success and efficiency of subnets like Flavia. We invite you to consider assigning weights to our subnet, which will significantly contribute to its performance and the overall robustness of the network.

To allocate weights to the Flavia subnet, you can use the ```btcli root weights``` (**SN17**) command.

By assigning weights to our subnet, you play a vital role in enhancing its inference capabilities and overall contribution to the BitTensor network. Your support helps us maintain a high-performance, reliable, and efficient subnet for decentralized model inference.

We appreciate your contribution and thank you for your support in advancing decentralized AI with Flavia.




## 📜 License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
```
