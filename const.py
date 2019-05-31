"""
Configuration and hyperparameters
"""
import torch
import numpy as np
import torchvision.transforms as transforms
import util
np.random.seed(0)
torch.manual_seed(0)

SAVE_PATH = 'save/' #+ util.get_run_name()
MODELS = ['NaiveConditionedCNN', 'PretrainedResNet', 'BranchedCOIL']

''' --- Config Settings --- '''
DATA_PATH = 'overfit_data/'
CURR_MODEL = MODELS[2]
AUGMENT_DATA = False
MODEL_WEIGHTS = SAVE_PATH + 'test_weights_340.pth'
EPOCHS = 100
SAVE_EVERY = 5

''' --- Constants --- '''
DRIVING_LOG_PATH = DATA_PATH + 'driving_log.csv'
USE_NORMALIZE = (CURR_MODEL == 'PretrainedResNet')
NORMALIZE_FN = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

NVIDIA_H, NVIDIA_W = 66, 200
CONFIG = {
    'batchsize': 64,
    'input_width': NVIDIA_W,
    'input_height': NVIDIA_H,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.05,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
    'bias': 0.8, # 1.0 is equivalent to not using this bias
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

CONTROLS = {0: 'Straight', 1: 'Left', 2: 'Right'}


