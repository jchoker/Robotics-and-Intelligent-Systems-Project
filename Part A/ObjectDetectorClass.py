import tensorrt as trt
from tensorrt_model import TRTModel
from ssd_tensorrt import load_plugins, parse_boxes,TRT_INPUT_NAME, TRT_OUTPUT_NAME
import numpy as np
import cv2
#import os
#import ctypes

#A full list of the detection class indices can be found in the following link
#https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_complete_label_map.pbtxt
BACKPACK_LABEL = 27
BOTTLE_LABEL = 44
KEYBOARD_LABEL = 76
PERSON_LABEL = 1

OBSTACLE_LABEL = -1

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])

def bgr8_to_ssd_input(camera_value):
    ''' convert camera bgr image to ssd input'''
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]

class ObjectDetector(object):
    
    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        load_plugins()
        self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME],output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
        self.preprocess_fn = preprocess_fn
        
    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return parse_boxes(trt_outputs)
    
    def __call__(self, *inputs):
        return self.execute(*inputs)        
        
    def get_tracked_label(self,image):
        ''' this method will check for single occurrences of the 4 tracked labels in the camera image
        and will return the corresponding label if any'''     
        imgsized= cv2.resize(image,(300,300))
        # compute all detected objects
        detections = self.execute(imgsized)
        total_detections_count =len(detections)
        
        print(f'total {total_detections_count}')
        #matching_detections = [d for d in detections[0] if d['label'] == int(label_widget.value)]
        
        bottle_detections = len([d for d in detections[0] if d['label'] == BOTTLE_LABEL])
        backpack_detections = len([d for d in detections[0] if d['label'] == BACKPACK_LABEL])
        keyboard_detections = len([d for d in detections[0] if d['label'] == KEYBOARD_LABEL])
        person_detections = len([d for d in detections[0] if d['label'] == PERSON_LABEL])

        if total_detections_count > 1:
            return OBSTACLE_LABEL # consider multi-object detection as obstacle        
        elif bottle_detections > 0:
            return BOTTLE_LABEL
        elif backpack_detections > 0:
            return BACKPACK_LABEL
        elif keyboard_detections > 0:
            return KEYBOARD_LABEL
        elif person_detections > 0:
            return PERSON_LABEL
        elif total_detections_count > 0:
            return OBSTACLE_LABEL 
        else:
            return None