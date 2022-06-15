"""
Check that the system works when looping through a single video and doing nothing
"""

from opencv_object_detector.runner import Detector

if __name__ == '__main__':

    model_root_path = "/home/alex/pretrained-models/yolov3-coco/"
    det = Detector(
        model_config_path = model_root_path+"yolov3.yolov3.cfg",
        model_weights_path = model_root_path+"yolov3.weights",
        data_labels_path = model_root_path+"coco.names",
        input_path = "/home/alex/project-videos/demo.mp4",

    )

    det.run_video()