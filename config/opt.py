import argparse

parser = argparse.ArgumentParser(description='Model options')

'''
----------------------------Pose model option--------------------------------------------------
'''

parser.add_argument('--pose_weight', default='weights/sppe/duc_se.pth', type=str,
                    help='Pose weight location')
parser.add_argument('--pose_backbone', default="seresnet101", type=str,
                    help='backbone')
parser.add_argument('--pose_cls', default=17, type=int,
                    help='sparse_decay')
parser.add_argument('--DUC_idx', default=0, type=int,
                    help='epoch of lr decay')
parser.add_argument('--pose_cfg', default=None, type=str,
                    help='The cfg')
parser.add_argument('--pose_batch', default=80, type=int,
                    help='The batch of pose')

parser.add_argument('--input_height', default=320, type=int,
                    help='input_height')
parser.add_argument('--input_width', default=256, type=int,
                    help='input_width')
parser.add_argument('--output_height', default=80, type=int,
                    help='output_height')
parser.add_argument('--output_width', default=64, type=int,
                    help='output_width')

'''
----------------------------Detection model option---------------------------------------------------------
'''
parser.add_argument('--confidence', default=0.05, type=float,
                    help='The confidence of yolo')
parser.add_argument('--num_classes', default=80, type=int,
                    help='Number of class of yolo')
parser.add_argument('--nms_thresh', default=0.33, type=float,
                    help='nms threshold')
parser.add_argument('--input_size', default=416, type=int,
                    help='Input size of yolo')
parser.add_argument('--yolo_cfg', default="config/yolo_cfg/yolov3.cfg", type=str,
                    help='Pose weight location')
parser.add_argument('--yolo_weight', default='weights/yolo/yolov3.weights', type=str,
                    help='backbone')

'''
----------------------------CNN model option---------------------------------------------------------
'''
parser.add_argument('--CNN_backbone', default="mobilenet", type=str,
                    help='The backbone of CNN')
parser.add_argument('--CNN_classes', default="freestyle,frog", type=str,
                    help='CNN model class (separated with ,)')
parser.add_argument('--CNN_weight', default=None, type=str,
                    help='Rhe weight of CNN')


'''
---------------------------Video option----------------------------------------
'''
parser.add_argument('--video_height', default=540, type=int,
                    help='output_height')
parser.add_argument('--video_width', default=720, type=int,
                    help='output_width')
parser.add_argument('--video_path', default="video/video_sample/video4_Trim.mp4",type=str,
                    help='output_width')
parser.add_argument('--out_video_path', default=None, type=str,
                    help='output_width')
parser.add_argument('--img_folder', default="img/test", help='output_width')

opt = parser.parse_args()

