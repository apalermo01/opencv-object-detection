import json
import cv2
import numpy as np
import logging
from datetime import datetime as dt
timestamp = dt.now().strftime('%Y%m%d-%H:%M:%S')
log_path = f"./logs/log_{timestamp}"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    filename=log_path,
                    format="%(asctime)s:%(levelname)s:%(message)s")

"""
TODO: 
- load video
- initialize results
- loop through video
- load the model
- 
"""

def run_detector_from_config(config_path):
    pass
class Detector:
    """
    load_from_json

    config_path

    input_path

    output_path


    reference articles: 
    https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9
    """
    def __init__(self,
                 model_config_path=None,
                 model_weights_path=None,
                 data_labels_path=None,
                 input_path=None,
                 input_size=608,
                 output_path=None,
                 show_raw_output=True,
                 show_model_output=True,
                 log_path=None,
                 conf_threshold=0.5,
                 nms_threshold=0.4,):
        
        self.model_config_path=model_config_path
        self.model_weights_path=model_weights_path
        self.data_labels_path=data_labels_path
        self.input_path=input_path
        self.output_path=output_path
        self.show_raw_output=show_raw_output
        self.show_model_output=show_model_output
        self.input_size=input_size
        self.conf_threshold=conf_threshold
        self.nms_threshold=nms_threshold
        # if log_path is None:
        #     timestamp = dt.now().strftime('%Y%m%d-%H:%M:%S')
        #     log_path = f"./logs/log_{timestamp}"
        #     logging.basicConfig(filename=log_path)

        # initialization
        self.load_labels()
        self.load_model()

    def load_labels(self):
        with open(self.data_labels_path, "r") as f:
            self.names = f.read().strip().split("\n")

        self.colors = np.random.uniform(0, 255, size=(len(self.names), 3))
        logger.info("labels loaded")
    
    def load_model(self):
        # self.net = cv2.dnn.readNetFromDarknet(self.model_weights_path, self.model_config_path)
        self.net = cv2.dnn.readNetFromDarknet(self.model_config_path,
                                   self.model_weights_path,)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        logger.info("model loaded")

    def run_video(self):
        self.cap = cv2.VideoCapture(self.input_path)
        logger.info("video loaded")
        inferences = dict()
        frame_num = 0
        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if ret:
                if self.show_raw_output:
                    cv2.imshow("output", frame)
                    cv2.waitKey(1)

                res = self.run_model(frame)
                inferences[frame_num] = res 

                if self.show_model_output:
                    self.show_output(frame, res)

            else:
                logger.error("frame read failed, exiting")
                break

            frame_num += 1

        with open(self.output_path, "w") as f:
            json.dump(inferences, f, indent=2)

    def run_model(self, frame):
        scale = 0.00392
        orig_w = frame.shape[1]
        orig_h = frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame,
                                     1/255,
                                     (self.input_size, self.input_size),
                                     #(0, 0, 0), 
                                     swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        #print("layer names = ", layer_names)
        #print("unconncected out layers: ", self.net.getUnconnectedOutLayers())
        #assert False
        #print(cv2.cuda.getCudaEnabledDeviceCount())
        #assert False
        
        if self.using_gpu():
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        else:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        layer_outputs = self.net.forward(output_layers)
        outputs = self.parse_outputs(layer_outputs, orig_w, orig_h)

        return outputs

    def parse_outputs(self, layer_outputs, orig_width, orig_height):
        
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        centers = []

        # for each detection from each output layer, get confidence, class id, bbox params,
        # and ignore weak detections

        for out in layer_outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0]*orig_width)
                    center_y = int(detection[1]*orig_height)
                    w = int(detection[2]*orig_width)
                    h = int(detection[3]*orig_height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    centers.append([center_x, center_y])

        # apply nms
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        # go through detections and save predictions to results
        results = {int(i): {'bbox': boxes[i],
                            'center_coord': centers[i],
                            'confidence': confidences[i],
                            'class_id': class_ids[i],
                            'class_name': self.names[class_ids[i]]}
                    for i in indices.flatten()}

        return results

    def show_output(self, frame, results):
        
        for res in results:
            out = results[res]
            label = out['class_name']
            color = self.colors[out['class_id']]
            bbox = out['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            print(f"x={x}; y={y}; w={w}; h={h}")
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            cv2.putText(frame, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 2)
        cv2.imshow('bbox output', frame)
        cv2.waitKey(1)
            
    def using_gpu(self):
        """
        https://stackoverflow.com/questions/61492452/how-to-check-if-opencv-is-using-gpu-or-not
        """
        count = cv2.cuda.getCudaEnabledDeviceCount()

        if count > 0:
            return True
        else:
            return False