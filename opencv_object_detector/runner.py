import json
import cv2
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

    """
    def __init__(self,
                 model_config_path=None,
                 model_weights_path=None,
                 data_labels_path=None,
                 input_path=None,
                 output_path=None,
                 show_raw_output=True,
                 show_model_output=True,
                 log_path=None,):
        
        self.model_config_path=model_config_path
        self.model_weights_path=model_weights_path
        self.data_labels_path=data_labels_path
        self.input_path=input_path
        self.output_path=output_path
        self.show_raw_output=show_raw_output
        self.show_model_output=show_model_output

        # if log_path is None:
        #     timestamp = dt.now().strftime('%Y%m%d-%H:%M:%S')
        #     log_path = f"./logs/log_{timestamp}"
        #     logging.basicConfig(filename=log_path)

        # initialization
        self.load_labels()

    def load_labels(self):
        with open(self.data_labels_path, "r") as f:
            self.names = f.read().strip().split("\n")
        logger.info("labels loaded")
    def run_video(self):
        self.cap = cv2.VideoCapture(self.input_path)
        logger.info("video loaded")
        while self.cap.isOpened():

            ret, frame = self.cap.read()

            if ret:
                if self.show_raw_output:
                    cv2.imshow("output", frame)
                    cv2.waitKey(1)

            else:
                logger.error("frame read failed, exiting")

