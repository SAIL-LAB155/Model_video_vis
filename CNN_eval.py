#-*- coding: utf-8 -*-
from config.opt import opt
import torch

import os
CNN_option = os.path.join("/".join(opt.CNN_weight.replace("\\", "/").split("/")[:-1]), "option.pth")
if os.path.exists(CNN_option):
    info = torch.load(CNN_option)
    opt.CNN_backbone = info.backbone
    opt.CNN_class = info.classes.split(",")

from src.human_detection_CNN import HumanDetection as ImgProcessor
import cv2
from config.config import frame_size


IP = ImgProcessor()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
gray = True if "GRAY" or "gray" in opt.yolo_weight else False


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
                IP.process_img(frame, gray=gray)
                img, black_img = IP.visualize()
                IP.classify(black_img)

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
