from opencv_object_detector.runner import Detector

if __name__ == '__main__':

    model_root_path = "/home/alex/pretrained-models/"
    model_name = "yolo-drone/"

    names = "drone.names"
    cfg = "yolo-drone.cfg"
    weights = "yolo-drone.weights"
    
    input_path = "/home/alex/project-videos/demo.mp4"

    det = Detector(
        model_config_path = model_root_path+model_name+cfg,
        model_weights_path = model_root_path+model_name+weights,
        data_labels_path = model_root_path+model_name+names,
        input_path = input_path,
        show_raw_output=False,
        show_model_output=True,
        input_size=416,
        output_path="./output/test",
        conf_threshold=0.5,
        nms_threshold=0.4,
    )

    det.run_video()