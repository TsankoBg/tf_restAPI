import os
os.urandom(24)

USER_DATA = {
    "robert": "seeitall"}

TENSOR_CONFIG = {
    'tensor': 'D:/TensorFlow/models/research/object_detection/',
    'NUM_CLASSES' : 92
}
GRAPHS_CONFIG={
    'mobielnetv1': 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb',
    'resnet101': 'ssd_mobilenet_v1_coco_2017_11_17/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb',
    'graph': "model/frozen_inference_graph.pb"
}
LABELS_CONFIG={
    'mscoco': 'data/mscoco_label_map.pbtxt',
    'custom': 'model/mscoco_label_map.pbtxt'
}
FLAGS={
    'graph':'imagenet/classify_image_graph_def.pb',
    'labelmap':'imagenet/imagenet_2012_challenge_label_map_proto.pbtxt',
    'humanlabel':'imagenet/imagenet_synset_to_human_label_map.txt'
}
valid_images = [".jpg", ".gif", ".png", ".tga"]
valid_URL_images = ["jpg", "gif", "png", "tga"]