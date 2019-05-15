from os.path import join
import torch
from torch.utils import data
from config import *
from load_data import preprocess
import csv
import numpy as np
import random
import cv2

class DrivingDataset(data.Dataset):
    def __init__(self, csv_driving_data, augment_data=True, data_dir='data'):
        with open(csv_driving_data, 'r') as f:
            data = [row for row in csv.reader(f)][1:]
        self.data = data
        self.augment_data = augment_data
        self.data_dir = 'data'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']
        ct_path, lt_path, rt_path, steer, throttle, brake, speed = self.data[index]

        steer = np.float32(steer)
        throttle = np.float32(throttle)
        delta_correction = CONFIG['delta_correction']
        camera = random.choice(['frontal', 'left', 'right'])
        if camera == 'frontal':
            frame = preprocess(cv2.imread(join(self.data_dir, ct_path.strip())))
            steer = steer
        elif camera == 'left':
            frame = preprocess(cv2.imread(join(self.data_dir, lt_path.strip())))
            steer = steer + delta_correction
        elif camera == 'right':
            frame = preprocess(cv2.imread(join(self.data_dir, rt_path.strip())))
            steer = steer - delta_correction

        if self.augment_data:
            # mirror images with prob=0.5
            if random.choice([True, False]):
                frame = frame[:, ::-1, :]
                steer *= -1.

            # perturb slightly steering direction
            steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

            # if color images, randomly change brightness
            if CONFIG['input_channels'] == 3:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
                frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)
        X = torch.as_tensor(frame) # shape (h, w, c)
        y_steer = torch.as_tensor(steer) # shape (1,)
        return X, y_steer


def main():
    csv_driving_data = 'data/driving_log.csv'
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]
    data = DrivingDataset(driving_data)
    print(len(data))
    X, y_steer = data[3]
    print(X.shape, type(X))
    print(y_steer.shape, type(y_steer))

if __name__ == '__main__':
    main()
