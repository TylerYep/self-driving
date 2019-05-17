import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import torch

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import os
import numpy as np
import const
from load_data import preprocess

from models import NaiveConditionedCNN
from resnet import PretrainedResNet
import torchvision.transforms as transforms

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

# TODO: move to const
shouldNormalize = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current high level command of the car
    high_level_control = data["high_level_control"]
    high_level_control = int(float(high_level_control))
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    # frames incoming from the simulator are in RGB format
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

    # perform preprocessing (crop, resize etc.)
    image_array = preprocess(frame_bgr=image_array)

    image_array = torch.as_tensor(image_array)
    if shouldNormalize:
        h, w, c = image_array.shape
        image_array = image_array.reshape((c, h, w)) # reshaped for normalize function
        image_array = normalize(image_array)
        image_array = image_array.reshape((h, w, c)) # reshaped back to expected shape

    # add singleton batch dimension
    #image_array = np.expand_dims(image_array, axis=0) # Shape (N, H, W, C)
    image_array = torch.unsqueeze(image_array, dim=0) # Shape(N, H, W, C)

    # Create measurements with speed and one-hot high-level control
    measurements = np.zeros((4, 1)) # 3 index is for speed, 0-2 index is one-hot high-level control
    measurements[3] = speed
    measurements[high_level_control] = 1
    measurements = torch.as_tensor(measurements)
    measurements = measurements.float()

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    #steering_angle = float(model(torch.tensor(image_array), measurements))
    outputs = float(model(image_array, measurements)) #outputs[0] is steer and outputs[1] is throttle
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    #throttle = 0.28
    
    steering_angle = outputs[0]
    throttle = outputs[1]
    print(steering_angle, throttle, high_level_control)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    # load model weights
    # weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
    model = NaiveConditionedCNN()
    #model = PretrainedResNet()
    print('Loading weights: {}'.format('save/test_weights.pth'))
    model.load_state_dict(torch.load('save/test_weights_80.pth'))
    model.eval()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
