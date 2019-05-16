"""
Configuration and hyperparameters
"""
DRIVING_LOG_PATH = 'data/driving_log.csv'
SAVE_PATH = 'save/'

NVIDIA_H, NVIDIA_W = 66, 200

CONFIG = {
    'batchsize': 32,
    'input_width': NVIDIA_W,
    'input_height': NVIDIA_H,
    'input_channels': 3,
    'delta_correction': 0.25,
    'augmentation_steer_sigma': 0.2,
    'augmentation_value_min': 0.2,
    'augmentation_value_max': 1.5,
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
    'crop_height': range(20, 140)
}