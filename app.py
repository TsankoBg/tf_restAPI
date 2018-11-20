import numpy as np
import cv2
from PIL.Image import Image
from PIL import Image
import matplotlib.pyplot
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('Agg')
from collections import defaultdict
import six.moves.urllib as urllib
from flask import Flask, jsonify, render_template
from flask import request
from flask import send_file
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)
# operator used for sorting
from operator import itemgetter
import sys
#import tarfile
import json
#from utils impor
import pytesseract
from pytesseract import image_to_string
import requests
import io
from io import BytesIO
from itertools import islice
from Object_Detector import ObjectDetector
from text_reader import ImageTextReader
import config

objectDetector=ObjectDetector()
imageTextReader= ImageTextReader()
app = Flask(__name__)
auth = HTTPBasicAuth()

@app.route("/")
def index():
    return render_template('index.html')

@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return config.USER_DATA.get(username) == password

@app.route('/scan/image/<path:imgid>')
@auth.login_required
def getObjectFromSingleImage(imgid):
    image=cv2.imread(imgid)
    return jsonify(objectDetector.scanImage(image)) 

@app.route('/scan/images/<path:fpath>')
@auth.login_required
def scanImagesFromFolder(fpath):
    return jsonify(objectDetector.scanImages(fpath))         

@app.route('/scan/url/<path:url>')
def scanFromURL(url):
    response = requests.get(url)
    #image = Image.open(BytesIO(response.content))
    #nparr = np.fromstring(BytesIO(response.content), np.uint8)
    #img_np = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
    file_bytes = np.asarray(bytearray(BytesIO(response.content).read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #cv2.imwrite('img4.jpg',imgage)
    return jsonify(objectDetector.scanImage(image))  

@app.route('/read/<path:img_path>')
@auth.login_required
def readText(img_path):
    image=cv2.imread(img_path)
    return imageTextReader.readText(image)

@app.route('/read/url/<path:url>')
def readTextURL(url):
    response = requests.get(url)
    #image = Image.open(BytesIO(response.content))
    file_bytes = np.asarray(bytearray(BytesIO(response.content).read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return imageTextReader.readText(image) 
    
@app.errorhandler(500)
def internal_error(error):
    return "Image not found"

#BACKGROUND WORK
@app.route("/submitted", methods=['POST'])
def upload():
    jsonData=[]
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")
        for img in uploaded_files:
            npimg = np.fromstring(img.read(), np.uint8)
            image = cv2.imdecode(npimg, 1)
            jsonData.append(objectDetector.scanImage(image))
            

    return jsonify(jsonData)   
    
if __name__ == "__main__":
    app.run()