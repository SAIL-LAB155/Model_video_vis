import torch
from config.opt import opt

img_folder = "img/test"

'''
------------------------------------------------------------------------------------------
'''

write_box = False
write_kps = False

device = "cuda:0"
print("Using {}".format(device))

track_idx = "all"    # If all idx, track_idx = "all"
track_plot_id = ["all"]   # If all idx, track_plot_id = ["all"]
assert track_idx == "all" or isinstance(track_idx, int)

plot_bbox = True
plot_kps = True
plot_id = True

resize_ratio = 0.5
show_size = (1280, 480)
store_size = (1280, 480)

