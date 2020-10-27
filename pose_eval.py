#-*- coding: utf-8 -*-
from config.opt import opt
from config import config
import torch
import cv2

import os
pose_option = os.path.join("/".join(opt.pose_weight.replace("\\", "/").split("/")[:-1]), "option.pkl")
pose_thresh = [0.05] * 17
pose_thresh.append((pose_thresh[-11] + pose_thresh[-12]) / 2)
opt.pose_thresh = pose_thresh
if os.path.exists(pose_option):
    info = torch.load(pose_option)
    opt.pose_backbone = info.backbone
    opt.pose_cfg = info.struct
    opt.pose_cls = info.kps
    opt.DUC_idx = info.DUC

    opt.output_height = info.outputResH
    opt.output_width = info.outputResW
    opt.input_height = info.inputResH
    opt.input_width = info.inputResW
    try:
        pose_thresh = list(map(lambda x: float(x), info.thresh.split(",")))
        pose_thresh.append((pose_thresh[-11] + pose_thresh[-12])/2)
        opt.pose_thresh = pose_thresh
    except:
        pass

print(opt.pose_thresh)

resize_ratio = config.resize_ratio
store_size = config.store_size
show_size = config.show_size

fourcc = cv2.VideoWriter_fourcc(*'XVID')
gray = True if "GRAY" in opt.yolo_weight or "gray" in opt.yolo_weight else False

from src.human_detection import HumanDetection


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resize_size = (int(self.width * resize_ratio), int(self.height * resize_ratio))
        self.IP = HumanDetection(self.resize_size)
        if opt.out_video_path:
            self.out = cv2.VideoWriter(opt.out_video_path, fourcc, 10, store_size)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, self.resize_size)
                kps, boxes, kps_score = self.IP.process_img(frame)
                img, img_black = self.IP.visualize()
                cv2.imshow("res", cv2.resize(img, show_size))

                cv2.waitKey(2)
                if opt.out_video_path:
                    self.out.write(cv2.resize(img, store_size))
            else:
                self.cap.release()
                if opt.out_video_path:
                    self.out.release()
                break


if __name__ == '__main__':
    VideoProcessor(opt.video_path).process_video()
