import torch
import torch.utils.data as data
from models import Model
import matplotlib.pyplot as plt
import const
from dataset import DrivingDataset
import cv2
import numpy as np
import loss_utils

def main():
    # load model weights
    model = Model(const.CURR_MODEL)
    print('Loading weights: {}'.format(const.MODEL_WEIGHTS))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(const.MODEL_WEIGHTS, map_location=device))
    model.eval()

    # obtain inputs and labels
    BATCH_SIZE = 1
    NUM_SHUFFLES = 18
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
    loss = outputs[high_level_controls][0][0] # steer prediction
    loss.backward()

    saliency = inputs.grad.data.abs()
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
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)


    plt.show()
if __name__ == '__main__':
    main()