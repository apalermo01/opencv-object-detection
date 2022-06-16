import json
import cv2
import numpy as np
import logging
from datetime import datetime as dt
from opencv_object_detector.modules.darknet_module import DarknetModule
timestamp = dt.now().strftime('%Y%m%d-%H:%M:%S')
log_path = f"./logs/log_{timestamp}"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    filename=log_path,
                    format="%(asctime)s:%(levelname)s:%(message)s")


class Detector:
    """
    model_config_path

    model_weights_path

    data_labels_path

    input_path

    input_size

    output_path

    show_raw_output

    show_model_output

    log_path
        <does not currently work>

    conf_threshold
    
    nms_threshold

    reference articles: 
    https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9
    """
    def __init__(self,
                 tracker_cfg,
                 input_path=None,
                 input_size=608,
                 output_path=None,
                 show_raw_output=True,
                 show_model_output=True,
                 log_path=None,
                 output_size=(1280, 720),
                 start_frame = 0,
                 end_frame = -1,
                 save_output_video = False,
                 draw_output_on_video = True):

        model_type = tracker_cfg['tracker_id']
        model_args = tracker_cfg['args']

        if model_type == 'darknet': 
            self.model = DarknetModule(**model_args)
        else:
            raise ValueError
        # parameters
        #self.model_config_path=model_config_path
        #self.model_weights_path=model_weights_path
        #self.data_labels_path=data_labels_path
        self.input_path=input_path
        self.output_path=output_path
        self.show_raw_output=show_raw_output
        self.show_model_output=show_model_output
        self.input_size=input_size
        self.draw_output_on_video = draw_output_on_video
        #self.conf_threshold=conf_threshold
        #self.nms_threshold=nms_threshold
        self.output_size=output_size
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.save_output_video = save_output_video

        if self.save_output_video:
            self.writer = None

    def run_video(self,
                  write_frame_on_output_video=True):
        """Run the pipeline

        :param write_frame_on_output_video:
        """
        self.cap = cv2.VideoCapture(self.input_path)
        logger.info("video loaded")
        inferences = dict()
        frame_num = 0

        # loop through all frames
        while self.cap.isOpened():
            

            ret, frame = self.cap.read()

            if frame_num < self.start_frame:
                frame_num += 1
                continue

            # if read was successful, then we run the pipeline on the frame
            if ret:
                res, frame = self.run_frame(frame,
                                             write_frame_on_output_video=write_frame_on_output_video,
                                             frame_id=frame_num)
                inferences[frame_num] = res

            else:
                logger.error("frame read failed, exiting")
                break

            if self.end_frame != -1 and frame_num > self.end_frame:
                logger.info("reached end frame, exiting")
                break

            frame_num += 1

        # save results to file
        if ".json" not in self.output_path:
            output_path = self.output_path + ".json"
        else:
            output_path = self.output_path
        with open(output_path + ".json", "w") as f:
            json.dump(inferences, f, indent=2)

    def run_frame(self, frame, write_frame_on_output_video=False, frame_id=-1):

        # optionally display the input image
        if self.show_raw_output:
            cv2.imshow("output", frame)
            cv2.waitKey(1)

        # run the model
        res = self.model.run(frame)

        # visualize output
        if self.draw_output_on_video or self.show_model_output or self.save_output_video:
            frame_drawn = self.model.draw_output(frame, res)
        if write_frame_on_output_video:
            frame_drawn = self.draw_frame_number_on_output_video(frame, frame_id)
        if self.show_model_output:
            frame_drawn = cv2.resize(frame_drawn, self.output_size)
            self.show_output(frame_drawn)
        if self.save_output_video:
            self.write_frame_to_file(frame_drawn)

        return res, frame

    def draw_frame_number_on_output_video(self,
                                          frame,
                                          frame_id):
        cv2.rectangle(frame, (0, 0), (200, 75), (100, 100, 100), -1)
        cv2.putText(frame,
                    text=f"frame: {frame_id}",
                    org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=1,)
        return frame

    def write_frame_to_file(self, frame_drawn,):
        if self.writer is None:
            if ".json" in self.output_path:
                output_video_path = self.output_path[:-5] + ".mp4"
            else:
                output_video_path = self.output_path + ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(output_video_path,
                                          fourcc,
                                          30,
                                          (self.output_size[1], self.output_size[0]))
        self.writer.write(frame_drawn)

    # def run_model(self, frame):
    #     scale = 0.00392
    #     orig_w = frame.shape[1]
    #     orig_h = frame.shape[0]
    #     blob = cv2.dnn.blobFromImage(frame,
    #                                  1/255,
    #                                  (self.input_size, self.input_size),
    #                                  #(0, 0, 0), 
    #                                  swapRB=True,
    #                                  crop=False)
    #     self.net.setInput(blob)
    #     layer_names = self.net.getLayerNames()
    #     if self.using_gpu():
    #         output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    #     else:
    #         output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    #     layer_outputs = self.net.forward(output_layers)
    #     outputs = self.parse_outputs(layer_outputs, orig_w, orig_h)

        # return outputs

    # def parse_outputs(self, layer_outputs, orig_width, orig_height):
        
    #     # initialization
    #     class_ids = []
    #     confidences = []
    #     boxes = []
    #     centers = []

    #     # for each detection from each output layer, get confidence, class id, bbox params,
    #     # and ignore weak detections
    #     for out in layer_outputs:
    #         for detection in out:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
    #             if confidence > self.conf_threshold:
    #                 center_x = int(detection[0]*orig_width)
    #                 center_y = int(detection[1]*orig_height)
    #                 w = int(detection[2]*orig_width)
    #                 h = int(detection[3]*orig_height)
    #                 x = center_x - w / 2
    #                 y = center_y - h / 2
    #                 class_ids.append(class_id)
    #                 confidences.append(float(confidence))
    #                 boxes.append([x, y, w, h])
    #                 centers.append([center_x, center_y])

    #     # apply nms
    #     indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

    #     # go through detections and save predictions to results
    #     print("len of indices = ", len(indices))
    #     if len(indices) > 0:
    #         results = {int(i): {'bbox': boxes[i],
    #                             'center_coord': centers[i],
    #                             'confidence': confidences[i],
    #                             'class_id': class_ids[i],
    #                             'class_name': self.names[class_ids[i]]}
    #                     for i in indices.flatten()}
    #     else:
    #         results = {}

    #     return results



    def show_output(self, frame):
        cv2.imshow('bbox output', frame)
        cv2.waitKey(1)
            
 