# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect,send_file
from gevent.pywsgi import WSGIServer

# deep learning utilities
from util import base64_to_pil
import keras
import torch
import cv2
import torch.nn as nn
import torchvision
#import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import net 
import numpy as np
from torchvision import transforms
# Declare a flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0




print('Model loaded. Start serving at http://127.0.0.1:5000/')


#routes

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/video_demo', methods=['GET'])
def video_demo():
    return render_template('video_demo.html')
    
    
@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.jpg")

        ########
        print("LOG:uploading of image works")
        #######

        #pre-processing
        data_hazy = cv2.imread("./uploads/image.jpg")
        data_hazy = cv2.cvtColor(data_hazy, cv2.COLOR_BGR2RGB)

        print("LOG:Reading of image works")
   
        data_hazy = (np.asarray(data_hazy)/255.0)
        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2,0,1)
        data_hazy = data_hazy.unsqueeze(0)
        dehaze_net =net.dehaze_net()
        dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth', map_location=torch.device('cpu')))
        clean_image = dehaze_net(data_hazy)
        torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "static/results/image.jpg")
        
        # Send Message
        res="Dehazed"
        # Serialize the result, you can add additional fields
        return jsonify(result=res)
        
    return render_template('predict.html')


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 8000), app)
    http_server.serve_forever()
