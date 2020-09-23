#-*- coding: utf-8 -*-
from config.opt import opt
import torch

import os
pose_option = os.path.join("/".join(opt.pose_weight.replace("\\", "/").split("/")[:-1]), "option.pkl")
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

from src.human_detection import HumanDetection as ImgProcessor
import cv2
from config.config import frame_size


IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
gray = True if "GRAY" in opt.yolo_weight or "gray" in opt.yolo_weight else False


class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if opt.out_video_path:
            self.out = cv2.VideoWriter(opt.out_video_path, fourcc, 15, frame_size)

    def process_video(self):
        cnt = 0
        while True:
            ret, frame = self.cap.read()
            cnt += 1
            if ret:
                frame = cv2.resize(frame, frame_size)
                kps, boxes, kps_score = IP.process_img(frame, gray=gray)
                img, black_img = IP.visualize()

                # tmp = cv2.resize(img, (720, 540))
                # cv2.imshow("res", tmp)
                cv2.imshow("res", img)
                cv2.waitKey(2)
                if opt.out_video_path:
                    self.out.write(img)
            else:
                self.cap.release()
                if opt.out_video_path:
                    self.out.release()
                break


if __name__ == '__main__':
    VideoProcessor(opt.video_path).process_video()
