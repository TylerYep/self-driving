import torch
import torch.utils.data as data
import torch.nn as nn
from models import Model
import matplotlib.pyplot as plt
import const
from dataset import DrivingDataset
import cv2
import numpy as np
import loss_utils
from scipy.misc import imresize

def main():
    # load model weights
    model = Model(const.CURR_MODEL)
    print('Loading weights: {}'.format(const.MODEL_WEIGHTS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(const.MODEL_WEIGHTS, map_location=device))
    model.eval()

    # obtain inputs and labels
    BATCH_SIZE = 3
    NUM_SHUFFLES = 23
    train_dataset = DrivingDataset(const.TRAIN_DRIVING_LOG_PATH)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    inputs, measurements, labels, high_level_controls = next(iter(train_dataloader))
    for i in range(NUM_SHUFFLES):
        inputs, measurements, labels, high_level_controls = next(iter(train_dataloader))


    # Display original pictures
    #plt.figure(1), plt.imshow(cv2.cvtColor(np.asarray(inputs[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))

    # Make the input require a gradient
    inputs.requires_grad_()

    inputs = inputs.to(device)
    labels = labels.to(device)
    measurements = measurements.to(device)

    outputs = model(inputs, measurements)

    #compute_saliency(outputs, inputs, high_level_controls)
    compute_activations(model, inputs, measurements, high_level_controls)

def compute_activations(model, inputs, measurements, high_level_controls):
    outputs, activations = model.forward_with_activations(inputs, measurements)
    cmap = plt.get_cmap('inferno')
    '''for activation in activations:
        activation = torch.abs(activation).mean(dim=1)[0].detach().numpy()
        activation /= activation.max()
        activation = cmap(activation)
        activation = np.delete(activation, 3, 2) # deletes 4th channel created by cmap
        activation = imresize(activation, [66, 200])
        plt.imshow(activation)
        plt.show()'''

    N = inputs.shape[0]
    inputs = inputs.detach().numpy()
    for i in range(N):
        plt.subplot(2, N, i+1)
        original = cv2.cvtColor(np.asarray(inputs[i]).astype('uint8'), code=cv2.COLOR_BGR2RGB)
        plt.imshow(original) 
        plt.axis('off')
        #plt.title('Input Images & First layer activations')
        plt.subplot(2, N, N + i + 1)
        activation = activations[0]
        activation = torch.abs(activation).mean(dim=1)[i].detach().numpy()
        activation /= activation.max()
        activation = cmap(activation)
        activation = np.delete(activation, 3, 2) # deletes 4th channel created by cmap
        activation = imresize(activation, [66, 200])
        plt.imshow(activation)
        #plt.imshow(activation[i], cmap=plt.cm.inferno)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def compute_saliency(outputs, inputs, high_level_controls):
    loss = outputs[high_level_controls][0][0] # steer prediction
    loss.backward()

    saliency = inputs.grad.data
    #saliency = -saliency # only for left turn
    saliency = saliency.abs()
    #saliency[saliency < 0] = 0

    saliency, index = torch.max(saliency, dim=3) # dim 3 is the channel dimension

   
    # Display saliency maps
    #plt.figure(2), plt.imshow(cv2.cvtColor(np.asarray(saliency[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))
    saliency = saliency.numpy()
    inputs = inputs.detach().numpy()
    N = inputs.shape[0]
    for i in range(N):
        plt.subplot(2, N, i+1)
        original = cv2.cvtColor(np.asarray(inputs[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB)
        plt.imshow(original) 
        plt.axis('off')
        plt.title(const.CONTROLS[int(high_level_controls[i].numpy())])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.gray)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

if __name__ == '__main__':
    main()