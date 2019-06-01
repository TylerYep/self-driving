from os.path import join
import torch
from torch.utils import data
import const
from load_data import preprocess, split_train_val
import numpy as np
import pandas as pd
import random
import cv2
import csv

import torchvision.transforms as transforms

class DrivingDataset(data.Dataset):
    ''' Uses the csv listed in const.py '''
    def __init__(self, csv_path=const.DRIVING_LOG_PATH):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h, w, c = const.CONFIG['input_height'], const.CONFIG['input_width'], const.CONFIG['input_channels']
        X, measurements, y_steer = None, None, None
        while True:
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

            if const.AUGMENT_DATA:
                # Mirroring images does not work well with high level controls
                # mirror images with prob=0.5

                # if random.choice([True, False]):
                #     frame = frame[:, ::-1, :]
                #     steer *= -1.

                # perturb slightly steering direction
                steer += np.random.normal(loc=0, scale=const.CONFIG['augmentation_steer_sigma'])

                # if color images, randomly change brightness
                if const.CONFIG['input_channels'] == 3:
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                    frame[:, :, 2] *= random.uniform(const.CONFIG['augmentation_value_min'], const.CONFIG['augmentation_value_max'])
                    frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

            # This augmentation technique ensures a more even distribution of steering vs straight driving
            # check that each element in the batch meet the condition
            steer_magnitude_thresh = np.random.rand()
            if (abs(steer) + const.CONFIG['bias']) < steer_magnitude_thresh:
                index = (index + 1) % len(self.data) # discard this element
            else:
                X = torch.as_tensor(frame) # shape (h, w, c)
                labels = torch.as_tensor([steer, throttle]) # shape (2,)
                #y_steer = torch.as_tensor(steer).unsqueeze(0) # shape (1,)
                measurements = np.zeros((4, 1)) # 3 index is for speed, 0-3 index is one-hot high-level control
                measurements[3] = speed
                measurements[high_level_control] = 1
                measurements = torch.as_tensor(measurements)
                if const.USE_NORMALIZE:
                    X = X.reshape((c, h, w)) # reshaped for normalize function
                    X = const.NORMALIZE_FN(X)
                    X = X.reshape((h, w, c)) # reshaped back to expected shape
                break

        return X, measurements.float(), labels, high_level_control

def main():
    data = DrivingDataset()
    print('Dataset length: ', len(data))
    X, measurements, labels, high_level_control = data[3]
    print(X.shape, type(X))
    print(labels.shape, type(labels))


    # Make sure you run clean_log.py first!!!
    # Uncomment the following to create a training and validation set split
    '''train_data, val_data = split_train_val(const.DRIVING_LOG_PATH, test_size=0.1)
    with open(const.VAL_DRIVING_LOG_PATH, "w", newline="") as f:
        cw = csv.writer(f, delimiter=",")
        cw.writerows(r for r in val_data)
    with open(const.TRAIN_DRIVING_LOG_PATH, "w", newline="") as f:
        cw = csv.writer(f, delimiter=",")
        cw.writerows(r for r in train_data)'''





if __name__ == '__main__':
    main()
