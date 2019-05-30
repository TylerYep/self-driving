import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

import const
from dataset import DrivingDataset
from models import Model
import loss_utils

from tensorboardX import SummaryWriter

def main():
    dataset = DrivingDataset()
    print("Dataset length: ", len(dataset))
    dataloaders = {
        'train': data.DataLoader(dataset, batch_size=const.CONFIG['batchsize'], shuffle=True, num_workers=8),
        'dev': data.DataLoader(dataset, batch_size=const.CONFIG['batchsize'], shuffle=False, num_workers=8),
        'test': data.DataLoader(dataset, batch_size=const.CONFIG['batchsize'], shuffle=False, num_workers=8)
    }

    model = Model(const.CURR_MODEL)
    #criterion = nn.MSELoss()
    criterion = loss_utils.branched_l2_loss
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model_trained = train_model(dataloaders, model, criterion, optimizer, num_epochs=const.EPOCHS)
    torch.save(model_trained.state_dict(), const.SAVE_PATH + 'test_weights_final.pth')

# Train
def train_model(dataloaders, model, criterion, optimizer, num_epochs=1):
    # Visualization on Tensboard
    tbx = SummaryWriter(const.SAVE_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            num_batches = 0
            for inputs, measurements, labels, high_level_controls in tqdm(dataloaders[phase]):
                num_batches += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                measurements = measurements.to(device)

                outputs = model(inputs, measurements)

                # only used for branched architecture
                high_level_control_masks = loss_utils.compute_branch_masks(high_level_controls, num_targets=2)

                loss = criterion(outputs, labels, high_level_control_masks) # only pass in masks for branched architecture

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
            running_loss = running_loss / num_batches
            tbx.add_scalar(phase + '/MSE', running_loss, epoch)
            print(phase + ":", running_loss)
            if ((epoch + 1) % const.SAVE_EVERY) == 0: # save every 20 epochs
                torch.save(model.state_dict(), const.SAVE_PATH + 'test_weights_' + str(epoch + 1) + '.pth')
    return model

if __name__ == '__main__':
    main()
