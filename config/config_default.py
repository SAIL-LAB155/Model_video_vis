import torch
from config.opt import opt


img_folder = "img/test"
frame_size = (opt.video_width, opt.video_height)

'''
------------------------------------------------------------------------------------------
'''

write_box = False
write_kps = False

device = "cuda:0"
print("Using {}".format(device))

confidence = 0.05
num_classes = 80
nms_thresh = 0.33
input_size = 416

track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True