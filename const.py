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
    'bias': 0.8,
    'crop_height': range(20, 140)
}