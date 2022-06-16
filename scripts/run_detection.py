from opencv_object_detector.runner import Detector

if __name__ == '__main__':

    model_root_path = "/home/alex/pretrained-models/"
    model_name = "yolo-drone-2/"

    # names = "drone.names"
    # cfg = "yolo-drone.cfg"
    # weights = "yolo-drone.weights"

    names = "drone.names"
    cfg = "drone.cfg"
    weights = "drone_best.weights"
    
    # input_path = "/home/alex/project-videos/demo.mp4"
    input_path = "/home/alex/project-videos/drone_demo.MTS"

    cfg = dict(
        tracker_id = 'darknet',
        args = dict(
            model_config_path = model_root_path+model_name+cfg,
            model_weights_path = model_root_path+model_name+weights,
            data_labels_path = model_root_path+model_name+names,
            conf_threshold=0.5,
            nms_threshold=0.4,
            input_size=608,
        )
    )
    det = Detector(
        tracker_cfg=cfg,
        input_path = input_path,
        show_raw_output=False,
        show_model_output=True,
        output_path="./output/test",
        save_output_video=True,
        start_frame=1500,
        end_frame=2500
    )

    det.run_video(write_frame_on_output_video=True)