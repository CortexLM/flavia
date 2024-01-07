import asyncio
import bittensor as bt
from template.protocol import TextToImage
from PIL import Image
from torchvision.transforms import ToPILImage

# Initialize Bittensor components
network_name = 'finney'

# User inputs for wallet details and UID
wallet_name = input("Enter your wallet name: ")
hotkey = input("Enter your hotkey: ")
uid = int(input("Enter the UID for querying: "))  # UID input

subtensor = bt.subtensor(network=network_name)
wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
dendrite = bt.dendrite(wallet=wallet)

def query_image(prompt, uid):
    # Refine the prompt

    metagraph = subtensor.metagraph(netuid=17)
    filtered_axon = metagraph.axons[uid]

    # Query Bittensor network
    responses = dendrite.query(
        [filtered_axon],
        TextToImage(prompt=prompt, model="dataautogpt3|OpenDalleV1.1", refiner=True),
        deserialize=False, 
        timeout=10
    )

    # Process the response to generate an image
    ts = responses[0].output[0]
    transform = ToPILImage()
    img = transform(bt.Tensor.deserialize(ts))
    return img

def main():
    user_prompt = input("Enter your prompt for image generation: ")
    generated_image = query_image(user_prompt, uid)

    # Save the generated image
    generated_image.save('./image-generated.png')
    print('The image has been successfully generated and saved as: image-generated.png')

if __name__ == "__main__":
    main()