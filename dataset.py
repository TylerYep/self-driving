from os.path import join
import torch
from torch.utils import data
import const
from load_data import preprocess
import csv
import numpy as np
import pandas as pd
import random
import cv2

class DrivingDataset(data.Dataset):
    def __init__(self, driving_log_csv, augment_data=True):
        self.data = pd.read_csv(driving_log_csv)
        self.augment_data = augment_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h, w, c = const.CONFIG['input_height'], const.CONFIG['input_width'], const.CONFIG['input_channels']
        ct_path, lt_path, rt_path, steer, throttle, brake, speed, high_level_control = self.data.iloc[index, :]

        steer = np.float32(steer)
        throttle = np.float32(throttle)
        high_level_control = np.int32(high_level_control)
        speed = np.float32(speed)

        delta_correction = const.CONFIG['delta_correction']
        camera = random.choice(['frontal', 'left', 'right'])
        if camera == 'frontal':
            frame = preprocess(cv2.imread(join(const.DATA_PATH, ct_path.strip())))
            steer = steer
        elif camera == 'left':
            frame = preprocess(cv2.imread(join(const.DATA_PATH, lt_path.strip())))
            steer = steer + delta_correction
        elif camera == 'right':
            frame = preprocess(cv2.imread(join(const.DATA_PATH, rt_path.strip())))
            steer = steer - delta_correction

        if self.augment_data:
            # mirror images with prob=0.5
            if random.choice([True, False]):
                frame = frame[:, ::-1, :]
                steer *= -1.

            # perturb slightly steering direction
            steer += np.random.normal(loc=0, scale=const.CONFIG['augmentation_steer_sigma'])

            # if color images, randomly change brightness
            if const.CONFIG['input_channels'] == 3:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                frame[:, :, 2] *= random.uniform(const.CONFIG['augmentation_value_min'], const.CONFIG['augmentation_value_max'])
                frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)
        X = torch.as_tensor(frame) # shape (h, w, c)
        y_steer = torch.as_tensor(steer).unsqueeze(0) # shape (1, )
        measurements = torch.zeros((4, 1)) # 0 index is for speed, 1-3 index is one-hot high-level control
        measurements[3] = speed
        measurements[high_level_control] = 1
        measurements = torch.as_tensor(measurements)
        return X, measurements.float(), y_steer

def main():
    data = DrivingDataset(const.DRIVING_LOG_PATH)
    print(len(data))
    X, measurements, y_steer = data[3]
    print(X.shape, type(X))
    print(y_steer.shape, type(y_steer))

if __name__ == '__main__':
    main()
