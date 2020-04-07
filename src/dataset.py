import os
import sys
import numpy as np
import pandas as pd
import random
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if 'google.colab' in sys.modules:
    DATA_PATH = '/content/'
else:
    DATA_PATH = 'all_data/lake_data/'

CLASS_LABELS = []
AUGMENT_DATA = False

CONFIG = {
    'batchsize': 64,
    'input_width': 200,
    'input_height': 66,
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


def get_collate_fn(device):
    def to_device(b):
        return list(map(to_device, b)) if isinstance(b, (list, tuple)) else b.to(device)
    return lambda x: map(to_device, default_collate(x))


def load_train_data(args, device, val_split=0.2):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    orig_dataset = DrivingDataset(DATA_PATH + 'driving_log.csv', transform=norm)
    if args.num_examples:
        n = args.num_examples
        data_split = [n, n, len(orig_dataset) - 2 * n]
        train_set, val_set = random_split(orig_dataset, data_split)[:-1]
    else:
        data_split = [int(part * len(orig_dataset)) for part in (1 - val_split, val_split)]
        train_set, val_set = random_split(orig_dataset, data_split)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn)
    return train_loader, val_loader, []


def load_test_data(args, device):
    norm = get_transforms()
    collate_fn = get_collate_fn(device)
    test_set = DrivingDataset(DATA_PATH + 'driving_log_test.csv', transform=norm)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             collate_fn=collate_fn)
    return test_loader


def get_transforms(img_dim=None):
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess(frame_bgr, verbose=False):
    """
    Perform preprocessing steps on a single bgr frame.
    These inlcude: cropping, resizing, eventually converting to grayscale
    :param frame_bgr: input color frame in BGR format
    :param verbose: if true, open debugging visualization
    :return:
    """
    if frame_bgr is None:
        print('\n\ndriving_log.csv links to invalid images!\n\n')
        print('Remember to run clean_log.py.')
        sys.exit()

    h, w = CONFIG['input_height'], CONFIG['input_width']
    frame_resized = cv2.resize(frame_bgr, dsize=(w, h))
    # eventually change color space
    # if CONFIG['input_channels'] == 1:
    #     frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(frame_bgr, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')


def load_img_file(img_path):
    return cv2.imread(os.path.join(DATA_PATH, img_path.strip()))


class DrivingDataset(Dataset):
    ''' Uses the csv listed in py '''
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']
        X, measurements, y_steer = None, None, None
        while True:
            ct_path, lt_path, rt_path, steer, throttle, brake, speed, high_level_control \
                = self.data.iloc[index, :]

            steer = np.float32(steer)
            throttle = np.float32(throttle)
            high_level_control = np.int32(high_level_control)
            speed = np.float32(speed)

            delta_correction = CONFIG['delta_correction']
            camera = random.choice(['frontal', 'left', 'right'])

            if camera == 'frontal':
                frame = preprocess(load_img_file(ct_path))
                steer = steer
            elif camera == 'left':
                frame = preprocess(load_img_file(lt_path))
                steer = steer + delta_correction
            elif camera == 'right':
                frame = preprocess(load_img_file(rt_path))
                steer = steer - delta_correction

            if AUGMENT_DATA:
                # Mirroring images does not work well with high level controls
                # mirror images with prob=0.5
                use_mirror = False
                if use_mirror: # LAKE_TRACK only
                    if random.choice([True, False]):
                        frame = frame[:, ::-1, :]
                        steer *= -1.

                # perturb slightly steering direction
                steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

                # if color images, randomly change brightness
                if CONFIG['input_channels'] == 3:
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                    frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'],
                                                     CONFIG['augmentation_value_max'])
                    frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                    frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

            # Augmentation technique ensures more even distribution of steering vs straight driving
            # check that each element in the batch meet the condition
            steer_magnitude_thresh = np.random.rand()
            if (abs(steer) + CONFIG['bias']) < steer_magnitude_thresh:
                index = (index + 1) % len(self.data) # discard this element
            else:
                X = torch.as_tensor(frame) # shape (h, w, c)
                labels = torch.as_tensor([steer, throttle]) # shape (2,)
                # y_steer = torch.as_tensor(steer).unsqueeze(0) # shape (1,)
                # 3 index is for speed, 0-3 index is one-hot high-level control
                measurements = np.zeros((4, 1))
                measurements[3] = speed
                measurements[high_level_control] = 1
                measurements = torch.as_tensor(measurements)
                if self.transform:
                    X = X.reshape((c, h, w)) # reshaped for normalize function
                    X = self.transform(X)
                    X = X.reshape((h, w, c)) # reshaped back to expected shape
                break

        return (X, measurements.float(), high_level_control), labels


def main():
    data = DrivingDataset(DATA_PATH + 'driving_log.csv')
    print('Dataset length: ', len(data))
    (X, measurements, high_level_control), labels = data[3]
    print(X.shape, type(X))
    print(labels.shape, type(labels))
    # Make sure you run clean_log.py first!!!


if __name__ == '__main__':
    main()
