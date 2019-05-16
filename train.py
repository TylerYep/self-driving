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
from models import NaiveConditionedCNN

def main():
    dataset = DrivingDataset(const.DRIVING_LOG_PATH)
    dataloaders = {
        'train': data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8), # TODO 32 on gpu
        'dev': data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8),
        'test': data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    }

    model = NaiveConditionedCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model_trained = train_model(dataloaders, model, criterion, optimizer)

    # Save
    torch.save(model_trained.state_dict(), 'save/test_weights.pth')
    print(model_trained)



# Train
def train_model(dataloaders, model, criterion, optimizer, num_epochs=3):
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

            for inputs, high_level_controls, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                high_level_controls = high_level_controls.to(device)

                outputs = model(inputs) # TODO , high_level_controls)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()# * inputs.size(0)
                print(running_loss)
                # running_corrects += torch.sum(preds == labels.data)

            # epoch_loss = running_loss / len(image_datasets[phase])
            # epoch_acc = running_corrects.double() / len(image_datasets[phase])
            #
            # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
            #                                             epoch_loss,
            #                                             epoch_acc))
    return model

if __name__ == '__main__':
    main()