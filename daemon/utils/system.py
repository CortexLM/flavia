# For GPU information, we will use the GPUtil library. 
# This requires the library to be installed. If not installed, we handle the import error.
import os
import platform
import sys
import psutil
from utils.logging import logging
import GPUtil

def get_gpu_info():
    """
    Get the GPU information if GPUtil is available. Filters GPUs based on CUDA_VISIBLE_DEVICES.
    """
    # Retrieve the list of all GPUs
    all_gpus = GPUtil.getGPUs()

    # Check if CUDA_VISIBLE_DEVICES is set and parse it
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        available_gpu_ids = [int(id) for id in cuda_visible_devices.split(',')]
        # Filter to get only those GPUs that are visible as per CUDA_VISIBLE_DEVICES
        available_gpus = [gpu for gpu in all_gpus if gpu.id in available_gpu_ids]
    else:
        # If CUDA_VISIBLE_DEVICES is not set, consider all GPUs as available
        available_gpus = all_gpus

    return [{"uuid": gpu.uuid, "name": gpu.name, "total_memory": f"{gpu.memoryTotal}", "free_memory": gpu.memoryFree, "memory_used": gpu.memoryUsed, "driver": gpu.driver, "load": gpu.load  } for gpu in available_gpus]

def display_system_info():
    logging.info("Gathering system information...")
    info = {
        "Operating System": platform.system(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": sys.version,
        "Total RAM": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "GPUs": get_gpu_info()
    }
    
    for key, value in info.items():
        logging.info(f"{key}: {value}")