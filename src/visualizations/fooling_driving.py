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

def make_fooling(inputs, measurements, labels, high_level_controls, inputs_fooling, target_steer, model):
    learning_rate = 5
    #plt.figure(1), plt.imshow(cv2.cvtColor(np.asarray(inputs[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))
    #plt.show()

    outputs = model(inputs_fooling, target_steer)
    loss = outputs[high_level_controls][0][0] # steer prediction
    print("Original steer prediction: ", loss)
    print("Original steer label: ", labels[0][0])
    #exit()
    while True:
        outputs = model(inputs_fooling, target_steer)
        loss = outputs[high_level_controls][0][0]
        print(loss)
        # If trying to make steer go up
        '''if loss > target_steer:
            print("Fooled steer prediction: ", loss)
            break'''
        # If trying to make steer go down
        if loss < target_steer:
            print("Fooled steer prediction: ", loss)
            break
        loss.backward()
        grads = inputs_fooling.grad.data
        inputs_fooling.data -= learning_rate * grads / grads.norm()
        inputs_fooling.grad.data.zero_()

    inputs = inputs.detach()
    inputs_fooling = inputs_fooling.detach()
    diff = np.asarray(inputs_fooling[0]).astype('uint8') - np.asarray(inputs[0]).astype('uint8')
    plt.figure(1), plt.imshow(cv2.cvtColor(np.asarray(inputs[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))
    plt.figure(2), plt.imshow(cv2.cvtColor(np.asarray(inputs_fooling[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))
    plt.figure(3), plt.imshow(cv2.cvtColor(diff, code=cv2.COLOR_BGR2RGB))
    plt.show()
    # Display inputs_fooling



def main():
    # load model weights
    model = Model(const.CURR_MODEL)
    print('Loading weights: {}'.format(const.MODEL_WEIGHTS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(const.MODEL_WEIGHTS, map_location=device))
    model.eval()

    # obtain inputs and labels
    BATCH_SIZE = 1
    NUM_SHUFFLES = 13
    train_dataset = DrivingDataset(const.TRAIN_DRIVING_LOG_PATH)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    inputs, measurements, labels, high_level_controls = next(iter(train_dataloader))
    for i in range(NUM_SHUFFLES):
        inputs, measurements, labels, high_level_controls = next(iter(train_dataloader))


    # Display original pictures
    #plt.figure(1), plt.imshow(cv2.cvtColor(np.asarray(inputs[0]).astype('uint8'), code=cv2.COLOR_BGR2RGB))

    # Make the input require a gradient
    inputs_fooling = inputs.clone()
    inputs_fooling = inputs_fooling.requires_grad_()

    inputs = inputs.to(device)
    labels = labels.to(device)
    measurements = measurements.to(device)

    target_steer = -0.5

    make_fooling(inputs, measurements, labels, high_level_controls, inputs_fooling, target_steer, model)


if __name__ == '__main__':
    main()