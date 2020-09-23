import os

src_sppe_folder = "weights/sppe"
video_output_folder = "output/sppe"
video_name = "video/ceiling/ceiling_long_video.mp4"

yolo_cfg = "weights/yolo/2/yolov3-original-1cls-leaky.cfg"
yolo_weight = "weights/yolo/2/best.weights"

os.makedirs(video_output_folder, exist_ok=True)

model_path = []
model_option_path = []
model_name = []

for model_fname in os.listdir(src_sppe_folder):
    if os.path.isdir(os.path.join(src_sppe_folder, model_fname)):
        model_folder = os.path.join(src_sppe_folder, model_fname)
        assert len(model_path) == len(model_option_path)
        for file_name in os.listdir(model_folder):
            file_path = os.path.join(model_folder, file_name)
            if "option" in file_name:
                model_option_path.append(file_path)
            elif "pkl" in file_name:
                model_path.append(file_path)
                model_name.append(file_name)


v_name = video_name.split("/")[-1][:-4]
for model, n in zip(model_path, model_name):
    out_video_name = os.path.join(video_output_folder, "{}-{}.avi".format(v_name, n[:-4]))
    cmd = "python pose_eval.py --pose_weight {} --video_path {} --out_video_path {} --yolo_weight {} --yolo_cfg {}".\
        format(model, video_name, out_video_name, yolo_weight, yolo_cfg)
    print(cmd)
    os.system(cmd)
