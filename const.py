"""
Configuration and hyperparameters
"""
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

DATA_PATH = 'overfit_data/'
SAVE_PATH = 'save/'
DRIVING_LOG_PATH = DATA_PATH + 'driving_log.csv'


NVIDIA_H, NVIDIA_W = 66, 200

CONFIG = {
    'batchsize': 32,
    'input_width': NVIDIA_W,
    'input_height': NVIDIA_H,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.05,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'crop_height': range(20, 140)
}

RESNET_CONFIG = {
    'batchsize': 32,
    'input_width': 224,
    'input_height': 600,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8,
    'crop_height': range(20, 140)
}