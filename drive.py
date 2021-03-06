import os
import base64
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import torch
import torchsummary

from PIL import Image, ImageOps
from flask import Flask, render_template
from io import BytesIO

from src.dataset import preprocess, get_transforms
from src import test

CONTROLS = {0: 'Straight', 1: 'Left', 2: 'Right'}
USE_NORMALIZE = False

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle, throttle, speed, high level command for the car
    steering_angle = data["steering_angle"]
    throttle = data["throttle"]
    speed = data["speed"]
    high_level_control = int(float(data["high_level_control"]))

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    # frames incoming from the simulator are in RGB format
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

    # perform preprocessing (crop, resize etc.)
    image_array = preprocess(image_array)

    image_array = torch.as_tensor(image_array)
    if USE_NORMALIZE:
        h, w, c = image_array.shape
        image_array = image_array.reshape((c, h, w)) # reshaped for normalize function
        image_array = get_transforms()(image_array)
        image_array = image_array.reshape((h, w, c)) # reshaped back to expected shape

    # add singleton batch dimension
    image_array = torch.unsqueeze(image_array, dim=0) # Shape(N, H, W, C)

    # Create measurements with speed and one-hot high-level control
    measurements = np.zeros((4, 1)) # 3 index is for speed, 0-2 index is one-hot high-level control
    measurements[3] = speed
    measurements[high_level_control] = 1
    measurements = torch.as_tensor(measurements, dtype=torch.float)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    # steering_angle = float(model(torch.tensor(image_array), measurements))
    outputs = model(image_array, measurements) # outputs[:,0] is steer and outputs[:,1] is throttle

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # throttle = 0.28

    if type(model).__name__ in ('BranchedCOIL', 'BranchedNvidia', 'BranchedCOIL_ResNet18'):
        steering_angle = float(outputs[high_level_control][:, 0])
        throttle = float(outputs[high_level_control][:, 1])
    else:
        steering_angle = float(outputs[:, 0])
        throttle = float(outputs[:, 1])

    # if np.abs(throttle) < 0.1:
    #     throttle = 0.0
    # if np.abs(steering_angle) < 0.1:
    #     steering_angle = 0.0
    # if np.abs(steering_angle) > 2.0: # 3.0
    #     throttle = 0.0

    send_control(steering_angle, throttle)
    print(steering_angle, throttle, CONTROLS[high_level_control])


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
    model = test()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
