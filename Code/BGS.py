import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy
from scipy import signal
import os
import skimage
from skimage import measure, morphology

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def BG_subtraction(input_video_path: str, output_bg_extact_path: str, output_bg_bin_path: str):
    #####
    # Adjustable params:
    med_fil_size = 15
    smooth_fil_size = 15
    smooth_fil_var = 0
    open_close_fil_size = 9
    num_of_iter = 2
    # min_obj_size =

    ###########
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)

    h, w = [params['height'],params['width']]
    frame_amount = params['frame_count']

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_bg = cv2.VideoWriter(output_bg_extact_path, fourcc, params['fps'], (params['width'], params['height']))
    out_bin = cv2.VideoWriter(output_bg_bin_path, fourcc, params['fps'], (params['width'], params['height']), isColor=False)

    # fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold)
    fgbg = cv2.createBackgroundSubtractorKNN()

    for i in tqdm(range(frame_amount-1)):
        ret, frame = vid.read()
        if ret:
            # if i>0:
            #     fgmask = fgbg.apply(frame, fgmask, 0.005)
            # else:
            #     fgmask = fgbg.apply(frame)
            # frame = cv2.medianBlur(frame, med_fil_size)
            # frame = cv2.GaussianBlur(frame, (smooth_fil_size, smooth_fil_size), smooth_fil_var)
            fgmask = fgbg.apply(frame)
            fgmask = cv2.medianBlur(fgmask, med_fil_size)
            fgmask = cv2.GaussianBlur(fgmask, (smooth_fil_size, smooth_fil_size), smooth_fil_var)

            # frame = cv2.medianBlur(frame, med_fil_size)
            # frame = cv2.GaussianBlur(frame, (smooth_fil_size, smooth_fil_size), smooth_fil_var)
            # fgmask = fgbg.apply(frame)
            ###############
            # kernel = np.ones((5, 5), np.uint8)
            # fgmask=cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel,iterations=2)

            fgmask[fgmask < 200] = 0

            ###############
            # # plt.imshow(fgmask,cmap='gray')
            # # plt.show()
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size, open_close_fil_size)),iterations=num_of_iter)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size, open_close_fil_size)),iterations=num_of_iter)
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size-2, open_close_fil_size-2)),iterations=num_of_iter)
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size-2, open_close_fil_size-2)),iterations=num_of_iter)
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size-4, open_close_fil_size-4)),iterations=num_of_iter)
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_close_fil_size-4, open_close_fil_size-4)),iterations=num_of_iter)
            # fgmask[fgmask<200] = 0
            fgmask[fgmask>=200] = 1
            fgmask = fgmask.astype(bool)

            # labels_mask = measure.label(fgmask)
            # regions = measure.regionprops(labels_mask)
            # regions.sort(key=lambda x: x.area, reverse=True)
            # if len(regions) > 1:
            #     for rg in regions[1:]:
            #         labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
            # labels_mask[labels_mask != 0] = 255
            # mask = labels_mask

            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (smooth_fil_size, smooth_fil_size))
            # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
            # # plt.imshow(fgmask,cmap='gray')
            # # plt.show()

            # assuming mask is a binary image
            # label and calculate parameters for every cluster in mask
            labelled = measure.label(fgmask)
            rp = measure.regionprops(labelled)

            # get size of largest cluster
            size = max([i.area for i in rp])

            # remove everything smaller than largest
            fgmask = skimage.morphology.remove_small_objects(fgmask, min_size=size, connectivity=1, in_place=True)
            # plt.imshow(fgmask,cmap='gray')
            # plt.show()
            fgmask = fgmask.astype(np.uint8)
            extracted_frame = np.stack([fgmask, fgmask, fgmask], axis=2) * frame
            fgmask = fgmask*255

            out_bin.write(fgmask)
            out_bg.write(extracted_frame)

    vid.release()
    out_bin.release()
    cv2.destroyAllWindows()

# class BackGroundSubtractor:
#     # When constructing background subtractor, we
#     # take in two arguments:
#     # 1) alpha: The background learning factor, its value should
#     # be between 0 and 1.
#     # 2) firstFrame: This is the first frame from the video
#     def __init__(self, alpha, firstFrame):
#         self.alpha = alpha
#         self.backGroundModel = firstFrame
#
#     def getForeground(self, frame):
#         # apply the background averaging formula:
#         # new_backround = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
#         self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)
#         return cv2.absdiff(self.backGroundModel.astype(np.uint8), frame)
#
# # some filtering before any further processing.
# def denoise(frame):
#     frame = cv2.medianBlur(frame, 5)
#     frame = cv2.GaussianBlur(frame, (5, 5), 0)
#
#     return frame
#
# def background_subtraction(name_video):
#
#     cap = cv2.VideoCapture(name_video)
#     w = int(cap.get(3))
#     h = int(cap.get(4))
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     video_binary = cv2.VideoWriter('video_binary_out.avi',fourcc,30,(w,h))
#
#     path1 = r'{}\frames'.format(os.getcwd())
#     path2 = r'{}\mask_frames'.format(os.getcwd())
#     os.makedirs(path1, exist_ok = True)
#     os.makedirs(path2, exist_ok = True)
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     if ret is True:
#         backSubtractor = BackGroundSubtractor(0.47, denoise(frame))
#         run = True
#     else:
#         run = False
#     counter = 0
#     while (run):
#         # Read a frame from the
#         ret, frame = cap.read()
#
#         # If the frame was properly read.
#         if ret is True:
#             # Show the filtered image
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#             cv2.imshow('input', denoise(frame))
#
#             # get the foreground
#             foreGround = backSubtractor.getForeground(denoise(frame))
#
#             # Apply thresholding on the background and do morpholigy opertion in order to get a better mask
#             ret, mask = cv2.threshold(foreGround, 10, 255, cv2.THRESH_BINARY)
#             mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
#             mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
#
#             # display the resulting mask
#             cv2.imshow('mask', mask)
#             cv2.imshow('mask2', mask2 * frame)
#
#             counter = counter + 1
#
#             cv2.imwrite(os.path.join(path1, 'frame_no_' + str(counter) +'.png'), frame)
#             cv2.imwrite(os.path.join(path2, 'mask_frame_no_' + str(counter) + '.png'), mask2)
#
#             mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
#             video_binary.write(mask2)
#
#             key = cv2.waitKey(10) & 0xFF
#         else:
#             break
#
#         if key == 27:
#             break
#
#     cap.release()
#     video_binary.release()
#     cv2.destroyAllWindows()
#     print('finish background subtraction')
#     return path1, path2, video_binary