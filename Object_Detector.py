from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import tensorflow as tf
from tensorflow.python.client import device_lib
from object_detection.utils import visualization_utils as vis_util
import json
import glob
import sys
import os
from io import BytesIO
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import numpy as np
import cv2
import config
from PIL import Image
from PIL.Image import Image
import matplotlib.pyplot
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('Agg')


class ObjectDetector:
    def __init__(self):
        if tf.__version__ < '1.8.0':
            raise ImportError(
                'Please upgrade your tensorflow installation to v1.8.* or later!')
        global detection_graph
        global label_map
        global category_index
        label_map = label_map_util.load_labelmap(
            config.LABELS_CONFIG['custom'])
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=92, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        detection_graph = self.getDetectionGraph()

    def getDetectionGraph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(config.GRAPHS_CONFIG['graph'], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.shape[0], image.shape[1]
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def run_inference_for_single_image(self, sess, image):
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

    def getImageName(self, iName):
        return iName.split('\\')[-1]

    def scanImage(self, image):
        with detection_graph.as_default():
            with tf.Session() as sess:
                detectionResult = []
                image_np_expanded = np.expand_dims(image, axis=0)
                output_dict = self.run_inference_for_single_image(sess, image)
                for indx, value in enumerate(output_dict['detection_scores']):
                    if value > 0.20:
                        humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                            'name')
                        detectionResult.append({  # 'image_name':str(image),
                            "class_ID": int(output_dict['detection_classes'][indx]),
                            "object": str(humanReadble),
                            "detection_score": float(value),
                            "object_cordinates": output_dict['detection_boxes'][indx].tolist(),
                        })
                sess.close()
        return detectionResult

    def scanImageDemo(self, image):
        with detection_graph.as_default():
            with tf.Session() as sess:
                detectionResult = []
                #image=self.load_image_into_numpy_array(image)
                #image_np_expanded = np.expand_dims(image, axis=0)
                output_dict = self.run_inference_for_single_image(sess, image)
                if output_dict['detection_scores'][0] > 0.3:
                    vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], category_index, instance_masks=output_dict.get(
                    'detection_masks'), use_normalized_coordinates=True, line_thickness=3, min_score_thresh=0.3)   
                for indx, value in enumerate(output_dict['detection_scores']):
                    if value > 0.20:
                        humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                            'name')
                        detectionResult.append({  # 'image_name':str(image),
                            "class_ID": str(output_dict['detection_classes'][indx]),
                            "object": str(humanReadble),
                            "detection_score": str(value),
                            "object_cordinates": str(output_dict['detection_boxes'][indx]),
                    })
                sess.close()
        return image

    def scanImages(self, fpath):
        files = []
        [files.extend(glob.glob(fpath + '/*' + e))
         for e in config.valid_images]
        imageItems = iter(enumerate(files))
        detectionResult = []
        with detection_graph.as_default():
            with tf.Session() as sess:
                for k, v in imageItems:
                    image = cv2.imread(v)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    output_dict = self.run_inference_for_single_image(
                        sess, image)
                    for indx, value in enumerate(output_dict['detection_scores']):
                        if value > 0.30:
                            humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                                'name')
                            detectionResult.append({"image_name": str(self.getImageName(v)),
                                                    "class_ID": int(output_dict['detection_classes'][indx]),
                                                    "object": str(humanReadble),
                                                    "detection_score": float(value),
                                                    "object_cordinates": output_dict['detection_boxes'][indx].tolist(),
                                                    })
            sess.close()
        return detectionResult

    def listFD(self, url, ext=''):
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(tuple(ext))]

    def scanImagesFromURL(self, url):
        detectionResult = []
        with detection_graph.as_default():
            with tf.Session() as sess:
                for file in self.listFD(url, config.valid_URL_images):
                    response = requests.get(file)
                    file_bytes = np.asarray(
                        bytearray(BytesIO(response.content).read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    output_dict = self.run_inference_for_single_image(
                        sess, image)
                    for indx, value in enumerate(output_dict['detection_scores']):
                        if value > 0.30:
                            humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                                'name')
                            detectionResult.append({"image_name": str(self.getImageName(file)),
                                                    "class_ID": int(output_dict['detection_classes'][indx]),
                                                    "object": str(humanReadble),
                                                    "detection_score": float(value),
                                                    "object_cordinates": output_dict['detection_boxes'][indx].tolist(),
                                                    })
            sess.close()
        return detectionResult

    def searchObjects(self, objects):
        objects = objects.upper()
        objects = objects.split(',')
        files = []
        [files.extend(glob.glob('img' + '/*' + e))
         for e in config.valid_images]
        imageItems = iter(enumerate(files))
        detectionResult = []
        with detection_graph.as_default():
            with tf.Session() as sess:
                for k, v in imageItems:
                    image = cv2.imread(v)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    output_dict = self.run_inference_for_single_image(
                        sess, image)
                    for indx, value in enumerate(output_dict['detection_scores']):
                        if value > 0.30:
                            humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                                'name')
                            if humanReadble.upper() in objects:
                                detectionResult.append({"image_name": str(self.getImageName(v)),
                                                        "class_ID": int(output_dict['detection_classes'][indx]),
                                                        "object": str(humanReadble),
                                                        "detection_score": float(value),
                                                        "object_cordinates": output_dict['detection_boxes'][indx].tolist(),
                                                        })
            sess.close()
        return detectionResult

    def searchObjectFromURL(self, url, objects):
        objects = objects.upper()
        objects = objects.split(',')
        detectionResult = []
        with detection_graph.as_default():
            with tf.Session() as sess:
                for file in self.listFD(url, config.valid_URL_images):
                    response = requests.get(file)
                    file_bytes = np.asarray(
                        bytearray(BytesIO(response.content).read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_np_expanded = np.expand_dims(image, axis=0)
                    output_dict = self.run_inference_for_single_image(
                        sess, image)
                    for indx, value in enumerate(output_dict['detection_scores']):
                        if value > 0.30:
                            humanReadble = category_index[output_dict['detection_classes'][indx]].get(
                                'name')
                            if humanReadble.upper() in objects:
                                detectionResult.append({"image_name": str(self.getImageName(file)),
                                                        "class_ID": int(output_dict['detection_classes'][indx]),
                                                        "object": str(humanReadble),
                                                        "detection_score": float(value),
                                                        "object_cordinates": output_dict['detection_boxes'][indx].tolist(),
                                                        })
            sess.close()
        return detectionResult

   # def detectAndLocateObjects(self,fphath):
