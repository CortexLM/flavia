import threading
import logging
import json
import os
import configparser
from huggingface_hub import snapshot_download
from utils.turbomind import TurboMind
from utils.sdfast import SDFast

class ModelManager:
    """
    A class to manage downloading, configuring, and running various machine learning models.
    """

    def __init__(self):
        self.threads = []
        self.models = {}
        self.base_directory = os.getcwd()
        self.config = self.load_config("config.json")
        self.load_models_from_config()
        self.initialize_models()

    def load_config(self, config_path):
        """
        Load the configuration file.

        Parameters:
        config_path (str): Path to the configuration file.
        """
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            logging.error(f"Configuration file {config_path} not found.")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from the config file {config_path}.")
            return {}

    def fetch_model(self, model_name):
        """
        Download a model snapshot from Hugging Face and save it to a specific directory.

        Parameters:
        model_name (str): Name of the model on Hugging Face.
        """
        model_folder = f"./models/{model_name.replace('/', '-')}/model"
        logging.debug(f'Fetching model {model_name}..')
        snapshot_download(repo_id=model_name, local_dir=model_folder)

    def load_models_from_config(self):
        """
        Load models as specified in the configuration.
        """
        models = self.config.get('models', {})
        self.load_diffusions(models.get('diffusions', []))
        self.load_turbomind(models.get('turbomind', []))

    def load_diffusions(self, diffusions):
        """
        Load diffusion models from the config.

        Parameters:
        diffusions (list): List of diffusion models to load.
        """
        for model_info in diffusions:
            model_name = model_info.get('modelName')
            if model_name:
                self.fetch_model(model_name)

    def load_turbomind(self, turbominds):
        """
        Load turbomind models from the config.

        Parameters:
        turbominds (list): List of turbomind models to load.
        """
        for model_info in turbominds:
            model_name = model_info.get('modelName')
            if model_name:
                self.fetch_model(model_name)

    def initialize_models(self):
        """
        Initialize specific models after loading them.
        """
        models = self.config.get('models', {})
        self.models["Qwen|Qwen-72B-Chat"] = TurboMind(self, model_path="/models/lmdeploy-llama2-chat-7b-w4/workspace", gpu_id=models["turbomind"][0]["gpu_id"], model_type=models["turbomind"][0]["modelType"])
        self.models["dataautogpt3|OpenDalleV1.1"] = SDFast(self, model_path="/models/dataautogpt3-OpenDalleV1.1/model", model_refiner="/models/stabilityai-stable-diffusion-xl-refiner-1.0/model", port=6001, model_type="t2i", gpu_id=models["diffusions"][0]["gpu_id"])