import config
from text_reader import ImageTextReader
from Object_Detector import ObjectDetector
from Object_Detector_ImageNet import ImagenetDetector
from itertools import islice
from io import BytesIO
import io
import requests
from pytesseract import image_to_string
import pytesseract
import json
import sys
from operator import itemgetter
from bs4 import BeautifulSoup
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)
from werkzeug.security import safe_str_cmp
from flask_jwt import JWT, jwt_required, current_identity
from flask_httpauth import HTTPBasicAuth
from flask_restful import Resource, Api
from flask import send_file
from flask import request
from flask import Flask, jsonify, render_template
import six.moves.urllib as urllib
from collections import defaultdict
import numpy as np
import cv2
from PIL.Image import Image
from PIL import Image
import matplotlib.pyplot
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('Agg')


objectDetector = ObjectDetector()
imagenet = ImagenetDetector()
imageTextReader = ImageTextReader()
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
    image = cv2.imread(imgid)
    return jsonify(objectDetector.scanImage(image))


@app.route('/scan/folder/<path:fpath>')
@auth.login_required
def scanImagesFromFolder(fpath):
    return jsonify(objectDetector.scanImages(fpath))


@app.route('/scan/url/image/<path:url>')
def scanFromURL(url):
    response = requests.get(url)
    file_bytes = np.asarray(
        bytearray(BytesIO(response.content).read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return jsonify(objectDetector.scanImage(image))


@app.route('/scan/url/images/<path:url>')
def scanImagesFromURL(url):
    return jsonify(objectDetector.scanImagesFromURL(url))


@app.route('/test/<path:url>')
def testinfsmoreg(url):
    return jsonify(imagenet.run_inference_on_image_ImageNet('img/' + url))


@app.route('/search/folder/<object_names>')
def searchObjects(object_names):
    return jsonify(objectDetector.searchObjects(object_names))


@app.route('/search/url/<path:url>/<object_names>')
def searchObjectsFromURL(url, object_names):
    return jsonify(objectDetector.searchObjectFromURL(url, object_names))


@app.route('/read/image/<path:img_path>')
@auth.login_required
def readText(img_path):
    image = cv2.imread(img_path)
    return imageTextReader.readText(image)


@app.route('/read/url/<path:url>')
def readTextURL(url):
    response = requests.get(url)
    file_bytes = np.asarray(
        bytearray(BytesIO(response.content).read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return imageTextReader.readText(image)


@app.route("/submitted", methods=['POST'])
def upload():
    jsonData = []
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file")
        for img in uploaded_files:
            npimg = np.fromstring(img.read(), np.uint8)
            image = cv2.imdecode(npimg, 1)
            jsonData.append(objectDetector.scanImage(image))

    return jsonify(jsonData)


@app.errorhandler(500)
def internal_error(error):
    return "Image not found"


if __name__ == "__main__":
    app.run()
