import torch
import cv2
import copy
import numpy as np
from config import config


from src.detector.yolo_detect import ObjectDetectionYolo
from src.detector.visualize import BBoxVisualizer
from src.tracker.track import ObjectTracker
from src.tracker.visualize import IDVisualizer
from config.opt import opt
from src.utils.img import torch_to_im, gray3D, cut_image_with_box
from src.CNNclassifier.inference import CNNInference

tensor = torch.FloatTensor


class HumanDetection:
    def __init__(self, show_img=True):
        self.object_detector = ObjectDetectionYolo(cfg=opt.yolo_cfg, weight=opt.yolo_weight)
        self.object_tracker = ObjectTracker()
        self.BBV = BBoxVisualizer()
        self.IDV = IDVisualizer(with_bbox=False)
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}
        self.show_img = show_img
        self.CNN_model = CNNInference()

    def init_sort(self):
        self.object_tracker.init_tracker()

    def clear_res(self):
        self.boxes = tensor([])
        self.boxes_scores = tensor([])
        self.img_black = np.array([])
        self.frame = np.array([])
        self.id2bbox = {}

    def visualize(self):
        self.img_black = cv2.imread('video/black.jpg')
        if config.plot_bbox and self.boxes is not None:
            self.frame = self.BBV.visualize(self.boxes, self.frame, self.boxes_scores)
            # cv2.imshow("cropped", (torch_to_im(inps[0]) * 255))
        if config.plot_id and self.id2bbox is not None:
            self.frame = self.IDV.plot_bbox_id(self.id2bbox, self.frame)
            # frame = self.IDV.plot_skeleton_id(id2ske, copy.deepcopy(img))
        return self.frame, self.img_black

    def process_img(self, frame, gray=False):
        self.clear_res()
        self.frame = frame

        with torch.no_grad():
            if gray:
                gray_img = gray3D(copy.deepcopy(frame))
                box_res = self.object_detector.process(gray_img)
            else:
                box_res = self.object_detector.process(frame)
            self.boxes, self.boxes_scores = self.object_detector.cut_box_score(box_res)

            if box_res is not None:
                self.id2bbox = self.object_tracker.track(box_res)

        return self.id2bbox

    def classify_whole(self):
        out = self.CNN_model.predict(self.img_black)
        idx = out[0].tolist().index(max(out[0].tolist()))
        pred = opt.CNN_class[idx]
        print("The prediction is {}".format(pred))

    def classify(self, frame):
        for box in self.id2bbox.values():
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            x2 = frame.shape[1] if x2 > frame.shape[1] else x2
            y2 = frame.shape[0] if y2 > frame.shape[0] else y2
            img = np.asarray(frame[y1:y2, x1:x2])
            # cv2.imshow("cut", img)
            # cv2.imwrite("img/tmp/0.jpg", img)
            out = self.CNN_model.predict(img)
            idx = out[0].tolist().index(max(out[0].tolist()))
            pred = opt.CNN_class[idx]
            print(pred)
            text_location = (int((box[0]+box[2])/2)), int((box[1])+50)
            cv2.putText(frame, pred, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)


IP = HumanDetection()
frame_size = (720, 540)


class VideoProcessor:
    def __init__(self, vp):
        self.cap = cv2.VideoCapture(vp)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, frame_size)
                IP.process_img(frame)
                img, img_black = IP.visualize()
                IP.classify_whole()
                cv2.imshow("res", img)
                cv2.waitKey(2)

            else:
                self.cap.release()
                break


if __name__ == '__main__':
    VideoProcessor(video_path).process_video()
