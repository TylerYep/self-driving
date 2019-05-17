import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data

import const
from dataset import DrivingDataset
from models import NaiveConditionedCNN, PretrainedResNet

from tensorboardX import SummaryWriter

def main():
    dataset = DrivingDataset(const.DRIVING_LOG_PATH, pretrain_normalize=False) # Set pretrain_normalize to true when using pretrained models
    print("Dataset length: ", len(dataset))
    dataloaders = {
        'train': data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8), # TODO 32 on gpu
        'dev': data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8),
        'test': data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    }

    model = NaiveConditionedCNN()
    #model = PretrainedResNet() # set pretrain_normalize=True in DrivingDataset
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model_trained = train_model(dataloaders, model, criterion, optimizer, num_epochs=100)

    # Save
    torch.save(model_trained.state_dict(), 'save/test_weights_final.pth')
    print(model_trained)

# Train
def train_model(dataloaders, model, criterion, optimizer, num_epochs=3):
    # Visualization on Tensboard
    tbx = SummaryWriter(const.SAVE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            for inputs, measurements, labels in dataloaders[phase]:
                num_batches += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                measurements = measurements.to(device)

                outputs = model(inputs, measurements)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()# * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
            running_loss = running_loss / num_batches
            tbx.add_scalar(phase + '/MSE', running_loss, epoch)
            print(phase + ":", running_loss)
            if ((epoch + 1) % 20) == 0: # save every 20 epochs
                torch.save(model.state_dict(), 'save/test_weights_' + str(epoch + 1) + '.pth')

            # epoch_loss = running_loss / len(image_datasets[phase])
            # epoch_acc = running_corrects.double() / len(image_datasets[phase])
            #
            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                             epoch_loss,
            #                                             epoch_acc))
    return model

if __name__ == '__main__':
    main()
