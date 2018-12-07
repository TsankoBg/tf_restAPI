import time
from threading import Thread
import config
from text_reader import ImageTextReader
from Object_Detector import ObjectDetector
#from Object_Detector_ImageNet import ImagenetDetector
from itertools import islice
from io import BytesIO
import io
import requests
from pytesseract import image_to_string
import json
import sys
import os
from operator import itemgetter
from bs4 import BeautifulSoup
from werkzeug.security import safe_str_cmp
from flask_httpauth import HTTPBasicAuth
from flask_restful import Resource, Api
from flask import Response, stream_with_context
from flask import send_file
from flask import request
from flask import Flask, jsonify, render_template, redirect, url_for
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


app = Flask(__name__)

auth = HTTPBasicAuth()

th = Thread()
finished = False


@app.route("/")
def index():
    return render_template('index.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/demo")
def demoRoute():
   # return redirect("loaded", code=302)
    return render_template('demo.html')


@auth.verify_password
def verify(username, password):
    """[This function checks if the username and password are correct]

    Arguments:
        username {[string]} -- [username]
        password {[string]} -- [description]

    Returns:
        [bool] -- [returns true if username and password are correct, false if username or password are wrong]
    """
    if not (username and password):
        return False
    return config.USER_DATA.get(username) == password


@app.route('/scan/image/<path:imgid>')
@auth.login_required
def getObjectFromSingleImage(imgid):
    """[This function scans a single image and returns the objects found]

    Arguments:
        imgid {[string]} -- [Image name located in local "img" folder]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """

    image = cv2.imread(imgid)
    return jsonify(objectDetector.scanImage(image))


@app.route('/scan/folder/<path:fpath>')
@auth.login_required
def scanImagesFromFolder(fpath):
    """[This function scans every image in a folder and returns the found objects]

    Arguments:
        fpath {[string]} -- [This is the name of the local folder, example "img" folder]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """

    return jsonify(objectDetector.scanImages(fpath))


@app.route('/scan/url/image/<path:url>')
def scanFromURL(url):
    """[This function reads and scans an image from url and return the found objects]

    Arguments:
        url {[string]} -- [A url of an image]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """
    imageTextReader = ImageTextReader()
    response = requests.get(url)
    file_bytes = np.asarray(
        bytearray(BytesIO(response.content).read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return jsonify(objectDetector.scanImage(image))


@app.route('/scan/url/images/<path:url>')
def scanImagesFromURL(url):
    """[This function scans a multiple images from given url and return the objects found]

    Arguments:
        url {[string]} -- [Url to a remote folder with  images]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """

    return jsonify(objectDetector.scanImagesFromURL(url))


# @app.route('/test/<path:url>')
# def testinfsmoreg(url):
#    imagenet = ImagenetDetector()
#    return jsonify(imagenet.run_inference_on_image_ImageNet('img/' + url))


@app.route('/search/folder/<object_names>')
def searchObjects(object_names):
    """[This function searches objects in local folder]

    Arguments:
        object_names {[array]} -- [Array of all objects to be searched]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """

    return jsonify(objectDetector.searchObjects(object_names))


@app.route('/search/url/<path:url>/<object_names>')
def searchObjectsFromURL(url, object_names):
    """[This function searchs for object in given url folder by calling objectDetector class]

    Arguments:
        url {[string]} -- [URL path to a folder with images]
        object_names {[Array]} -- [Arra of all objects to be searched]

    Returns:
        [json] -- [Array of found objects, image name, object found, accuracy, boxes]
    """

    return jsonify(objectDetector.searchObjectFromURL(url, object_names))


@app.route('/read/image/<path:img_path>')
@auth.login_required
def readText(img_path):
    """[This method calls imageTextReader to read a text from given image]

    Arguments:
        img_path {[string]} -- [path to image]

    Returns:
        [string] -- [Found text]
    """
    imageTextReader = ImageTextReader()
    image = cv2.imread(img_path)
    return imageTextReader.readText(image)


@app.route('/read/url/<path:url>')
def readTextURL(url):
    imageTextReader = ImageTextReader()
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


@app.route("/demoSubmitted", methods=['POST'])
def demoPOST():
    if request.method == 'POST':
        global th
        global finished
        finished = False
        file = Image.open(request.files['file'].stream)
        open_cv_image = np.array(file)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        try:
            os.remove('static/img/newImageUsed.jpg')
        except OSError:
            pass

        th = Thread(target=something, args=(open_cv_image,))
        th.start()
    return render_template('loading.html')


def something(file1):
    """ The worker function """
    global finished
    img = objectDetector.scanImageDemo(file1)
    cv2.imwrite('static/img/newImageUsed.jpg', img)
    time.sleep(10)
    finished = True


@app.route('/result')
def result():
    """ Just give back the result of your heavy work """
    return render_template("imagePage.html")
    # return redirect(url_for('static', filename="img/newImageUsed.jpg"))


@app.route('/status')
def thread_status():
    """ Return the status of the worker thread """
    return jsonify(dict(status=('finished' if finished else 'running')))


if __name__ == "__main__":
    app.run(Thread=True)
