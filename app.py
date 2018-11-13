from flask import Flask, jsonify, render_template
from flask_restful import Resource, Api
from flask_httpauth import HTTPBasicAuth
from flask_jwt import JWT, jwt_required, current_identity
from werkzeug.security import safe_str_cmp
import numpy as np
import cv2
import os
import datetime
import six.moves.urllib as urllib
# operator used for sorting
from operator import itemgetter
import sys
#import tarfile
import tensorflow as tf
import zipfile
from os import walk
import io
import glob
import matplotlib.pyplot
import json
import jwt
matplotlib.pyplot.switch_backend('Agg')
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL.Image import Image
from object_detection.utils import label_map_util
#from utils impor
import pytesseract
import requests
from io import BytesIO
from PIL import Image
from pytesseract import image_to_string
from itertools import islice
from itsdangerous import (TimedJSONWebSignatureSerializer
                          as Serializer, BadSignature, SignatureExpired)
# import random
import config
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
if tf.__version__ < '1.8.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')
from tensorflow.python.client import device_lib

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

label_map = label_map_util.load_labelmap( config.LABELS_CONFIG['custom'])
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=92, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def getDetectionGraph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.GRAPHS_CONFIG['graph'], 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

detection_graph = getDetectionGraph()

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape[0], image.shape[1]
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(sess,image):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {
        output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph(
            ).get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                   real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                   real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.6), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={
                           image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(
        np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

#Remove path and return only image name
def getImageName(iName):
    return iName.split('\\')[-1]

@app.route('/scan/image/<path:imgid>')
@auth.login_required
def getObjectFromSingleImage(imgid):
    image=cv2.imread(imgid)
    with detection_graph.as_default():
        with tf.Session() as sess:
            detectionResult=[]
            image_np_expanded = np.expand_dims(image, axis=0)
            output_dict = run_inference_for_single_image(sess,image)
            for indx,value in enumerate(output_dict['detection_scores']):
                if value > 0.60:
                    #print('detection score is '  + str(value)  + '-  class is '  + str(output_dict['detection_classes'][indx]))
                        #print('cordinates: ' + str(output_dict['detection_boxes'][indx]))
                    humanReadble= category_index[output_dict['detection_classes'][indx]].get('name')
                    detectionResult.append({  'image_name':str(imgid),
                                                'class_ID:':str(output_dict['detection_classes'][indx]),
                                                'object:': str(humanReadble),
                                                'detection_score':str(value),
                                                'object_cordinates':str(output_dict['detection_boxes'][indx]),
                                              })
    return jsonify(detectionResult) 

@app.route('/scan/images/<path:fpath>')
@auth.login_required
def scanImagesFromFolder(fpath):
    detection_graph = getDetectionGraph()
    valid_images = [".jpg",".gif",".png",".tga"]
    files = []
    [files.extend(glob.glob(fpath + '/*' + e)) for e in valid_images]
    imageItems=iter(enumerate(files))
    detectionResult=[]
    with detection_graph.as_default():
        with tf.Session() as sess:
            for k,v in imageItems:
                image=cv2.imread(v)
                image_np_expanded = np.expand_dims(image, axis=0)
                output_dict = run_inference_for_single_image(sess,image)
                for indx,value in enumerate(output_dict['detection_scores']):
                    if value > 0.30:
                        humanReadble= category_index[output_dict['detection_classes'][indx]].get('name')
                        detectionResult.append({'image_name':str(getImageName(v)),
                                                'class_ID:':str(output_dict['detection_classes'][indx]),
                                                'object:': str(humanReadble),                       
                                                'detection_score':str(value),
                                                'object_cordinates':str(output_dict['detection_boxes'][indx]),
                                                })
    return jsonify(detectionResult)         

@app.route('/scan/url/<path:url>')
def product(url):
    #print(url)
    detection_graph = getDetectionGraph()
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    with detection_graph.as_default():
        with tf.Session() as sess:
            detectionResult=[]
            image_np_expanded = np.expand_dims(image, axis=0)
            output_dict = run_inference_for_single_image(sess,image)
            for indx,value in enumerate(output_dict['detection_scores']):
                if value > 0.60:
                    #print('detection score is '  + str(value)  + '-  class is '  + str(output_dict['detection_classes'][indx]))
                        #print('cordinates: ' + str(output_dict['detection_boxes'][indx]))
                    humanReadble= category_index[output_dict['detection_classes'][indx]].get('name')
                    detectionResult.append({'class_ID:':str(output_dict['detection_classes'][indx]),
                                                'object:': str(humanReadble),
                                                'detection_score':str(value),
                                                'object_cordinates':str(output_dict['detection_boxes'][indx]),
                                                #'image_name':str(url),
                                                })
    return jsonify(detectionResult) 

@app.route('/read/<path:img_path>')
@auth.login_required
def readText(img_path):
     # Read image with opencv
    img = cv2.imread(img_path)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # Write image after removed noise
    #cv2.imwrite(src_path + "removed_noise.png", img)
    #  Apply threshold to get image with only black and white
    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    # Write the image after apply opencv to do some ...
    # cv2.imwrite(src_path + "thres.png", img)
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img)
    return str(result)

@app.errorhandler(500)
def internal_error(error):
    return "500 error"

if __name__ == "__main__":
    app.run()