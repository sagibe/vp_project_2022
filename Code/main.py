import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import os
from os.path import dirname, abspath
from pathlib import Path
from stabilization import stabilize_vid, Video_Stabilization
from BGS import BG_subtraction
from matting import matting_tracking



#######
# DIRECTORIES
base_dir = dirname(dirname(abspath(__file__)))
input_dir = os.path.join(base_dir, 'Inputs')
output_dir = os.path.join(base_dir, 'Outputs')

# STABILIZATION
# window_size = 5
# max_iter = 5
# num_lvls = 5

# input_stab_path = os.path.join(input_dir, 'INPUT.avi')
# output_stab_path = os.path.join(output_dir, 'stabilize_test.avi')
#
# stabilize_vid(input_stab_path, output_stab_path)
# Video_Stabilization(input_stab_path)

# BACKGROUND SUBTRACTION
# input_bg_path = os.path.join(output_dir, 'stabilize_1.avi')
# output_bg_extact_path = os.path.join(output_dir, 'extracted.avi')
# output_bg_bin_path = os.path.join(output_dir, 'binary.avi')
#
# BG_subtraction(input_bg_path, output_bg_extact_path,output_bg_bin_path)
# background_subtraction(input_bg_path)

# MATTING
new_bg_path = os.path.join(input_dir, 'background.jpg')
input_stab_path = os.path.join(output_dir, 'stabilize_1.avi')
input_fgbg_mask = os.path.join(output_dir, 'binary.avi')

matting_tracking(input_stab_path, input_fgbg_mask, new_bg_path)




print('hi')
