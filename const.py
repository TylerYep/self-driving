RUN_ON_GPU = True
"""
Configuration and hyperparameters
"""
import torch
import numpy as np
import torchvision.transforms as transforms
import util
np.random.seed(0)
torch.manual_seed(0)

SAVE_PATH = 'save/'
LOG_PATH = SAVE_PATH + util.get_run_name() + '/'
MODELS = ['NaiveConditionedCNN', 'PretrainedResNet', 'BranchedCOIL', 'BranchedNvidia', 'BranchedCOIL_ResNet18']

''' --- Config Settings --- '''
DATA_PATH = 'slow_data/'
CURR_MODEL = MODELS[3]
AUGMENT_DATA = (CURR_MODEL == 'NaiveConditionedCNN')
MODEL_WEIGHTS = SAVE_PATH + 'weights_80.pth' # 60 was p good on both, 0.0 at zero help, 100 even better
if RUN_ON_GPU:
    EPOCHS = 1000
    SAVE_EVERY = 5
else:
    EPOCHS = 100
    SAVE_EVERY = 5

''' --- Constants --- '''
DRIVING_LOG_PATH = DATA_PATH + 'driving_log.csv'
TRAIN_DRIVING_LOG_PATH = DATA_PATH + 'driving_log_train.csv'
VAL_DRIVING_LOG_PATH = DATA_PATH + 'driving_log_val.csv'
USE_NORMALIZE = (CURR_MODEL == 'PretrainedResNet')
NORMALIZE_FN = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

NVIDIA_H, NVIDIA_W = 66, 200
CONFIG = {
    'batchsize': 64,
    'input_width': NVIDIA_W,
    'input_height': NVIDIA_H,
    'input_channels': 3,
    'delta_correction': 0.15,
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


