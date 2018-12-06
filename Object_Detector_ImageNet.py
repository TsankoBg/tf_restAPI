
import re
import sys
import tarfile
from PIL import Image
from numpy import array
import numpy as np
from six.moves import urllib
import tensorflow as tf
import config
from NodeLookUP import NodeLookUp


class ImagenetDetector:
    def __init__(self):
        if tf.__version__ < '1.8.0':
            raise ImportError(
                'Please upgrade your tensorflow installation to v1.8.* or later!')

    def create_graph(self):
        with tf.gfile.FastGFile( config.FLAGS['graph'], 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


    def run_inference_on_image_ImageNet(self,image):
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        #img = Image.open("input.png")
        #print(tf.FastGFile(image))
        #image_data = array(image)
        # Creates graph from saved GraphDef.
        self.create_graph()
        detectionResult = []
        with tf.Session() as sess:
            # Some useful tensors:
            # 'softmax:0': A tensor containing the normalized prediction across
            #   1000 labels.
            # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
            #   float description of the image.
            # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
            #   encoding of the image.
            # Runs the softmax tensor by feeding the image_data as input to the graph.
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup=NodeLookUp()

            top_k = predictions.argsort()[-5:][::-1]
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                detectionResult.append({"human_string":str(human_string),
                "score":str(score),
                })
                #print('%s (score = %.5f)' % (human_string, score))
            return detectionResult

