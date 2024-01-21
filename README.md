<div align="center">

# **🌸 Flavia - Model Inference Subnet**
Execute large-scale models on the Bittensor network

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

#
> [!WARNING]  
> This is an beta version. Please report any bugs or suggest any improvements.

### 📋 Introduction
The Flavia project is a sub-network of the Bittensor network, focused on decentralized model inference at different scales. It represents a groundbreaking innovation in the field of distributed computing, bringing a new level of efficiency and scalability to model inference tasks. This project is not just a technical achievement; it's a visionary step towards a more interconnected and collaborative digital ecosystem.

Within the Flavia subnet, each node contributes to the overall inference process, making it a highly resilient and robust system. This decentralized approach ensures that the network is not reliant on any single point of failure, thereby significantly enhancing its reliability and uptime. Moreover, the Flavia project is designed to be adaptive, capable of handling various types of computational loads, from small-scale individual requests to large, complex queries that require more substantial computational resources.

One of the most striking aspects of the Flavia project is its commitment to democratizing access to advanced model inference capabilities. By distributing the computational load across a network of nodes, Flavia allows individuals and organizations with limited resources to benefit from high-quality model inference without the need for significant hardware investments. This aspect is particularly crucial in an era where data-driven decision-making is becoming increasingly prevalent across various sectors.
## 🔧 Setup

**NOTICE :** The execution of Flavia requires the installation of Sense for model inference, which is mandatory for miners and validators. Sense must be on a different server; to install it, <a href="https://github.com/CortexLM/sense">click here</a>.
> [!WARNING]  
> We strongly recommend the utilization of Python environments for mining/validating activities. This approach is crucial because code originating from various subnets can often experience packet conflicts when interacting with each other. By using isolated Python environments, developers can ensure that each subnet operates within its own dedicated space.

> [!NOTE]  
> We recommend the use of a local Subtensor.
### Installation
**Clone the Repository & install the necessary requirements**

````
git clone https://github.com/CortexLM/flavia.git && cd flavia && pip install -e .
````



**Mining:**

````
pm2 start neurons/miner/miner.py --interpreter python3 --name miner<ID>-net17 -- --netuid 17 --wallet.name <coldkey> --wallet.hotkey <hotkey> --sense.base_url http://host:port --sense.api_key (your api key) --logging.debug

pm2 start run_update_all.py --name auto_update_mining (for auto updater)
````

**Validator:**

````
pm2 start run_validator.py --name validator-net17 --interpreter python3 -- --process_name validator-net17 --netuid 17 --wallet.name <coldkey> --wallet.hotkey <hotkey> --sense.base_url http://<HOST>:<EXTERNAL_PORT> --sense.api_key <YOUR API KEY> --logging.debug
````

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
