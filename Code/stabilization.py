import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy
from scipy import signal
import os

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

def sift_point_matcher(I1,I2):
    num_of_pnts = 50
    minDistance = 5
    thresh = max((minDistance)**3, minDistance*100)

    # feat_pnts = cv2.goodFeaturesToTrack(I1,
    #                                  maxCorners=100,
    #                                  qualityLevel=0.01,
    #                                  minDistance=minDistance,
    #                                  blockSize=3)
    #
    # feat_pnts = np.squeeze(feat_pnts.astype(int)).T

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(I1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(I2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # img3 = cv2.drawMatches(I1, keypoints_1, I2, keypoints_2, matches[:1], I2, flags=2)
    # plt.imshow(img3)
    # plt.show()

    prev_pnts = np.zeros((num_of_pnts,2))
    cur_pnts = np.zeros((num_of_pnts, 2))
    for i in range(num_of_pnts):
        cur_match = matches[i]
        prev_pnts[i] = keypoints_1[cur_match.queryIdx].pt
        cur_pnts[i] = keypoints_2[cur_match.trainIdx].pt
    return prev_pnts, cur_pnts

# def smooth(trajectory):
#     smoothed_trajectory = np.copy(trajectory)
#     # Filter the x, y and angle curves
#     for i in range(3):
#         smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=50)
#
#     return smoothed_trajectory
#
# def fixBorder(frame):
#     s = frame.shape
#     # Scale the image 4% without moving the center
#     T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
#     frame = cv2.warpAffine(frame, T, (s[1], s[0]))
#     return frame

# def movingAverage(curve, radius):
#     window_size = 2 * radius + 1
#     # Define the filter
#     f = np.ones(window_size) / window_size
#     # Add padding to the boundaries
#     curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
#     # Apply convolution
#     curve_smoothed = np.convolve(curve_pad, f, mode='same')
#     # Remove padding
#     curve_smoothed = curve_smoothed[radius:-radius]
#     # return smoothed curve
#     return curve_smoothed

def stabilize_vid(input_video_path: str, output_video_path: str):
    """Stabilize input video and saves it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.

    Returns:
        None.
    """
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output = cv2.VideoWriter(output_video_path, fourcc, params['fps'], (params['width'], params['height']))

    ret, frame = vid.read()
    output.write(frame)
    first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    prev_frame_gray = first_frame_gray
    h, w = [params['height'],params['width']]

    frame_amount = params['frame_count']
    # frame_amount = 30

    # Intizalize tranformation matrix parameters
    cumulative_tx = 0
    cumulative_ty = 0
    cumulative_ang = 0
    cumulative_s = 1

    for i in tqdm(range(frame_amount-1)):
        ret, frame = vid.read()
        if ret:
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            prev_pnts, cur_pnts = sift_point_matcher(prev_frame_gray,cur_gray_frame)
            cur_m = cv2.estimateAffinePartial2D(cur_pnts, prev_pnts)  # will only work with OpenCV-3 or less

            # Translation parameters
            cumulative_tx+=cur_m[0][0, 2]
            cumulative_ty+=cur_m[0][1, 2]

            # Rotation angle parameter
            cumulative_ang+=np.arctan2(cur_m[0][1, 0], cur_m[0][0, 0])

            # # scale parameter
            # cumulative_s*= m[0][0, 0]/(np.cos(np.arctan2(m[0][1, 0], m[0][0, 0])))

            # Transformation matrix
            cumulative_m = np.array([[np.cos(cumulative_ang)*cumulative_s,-np.sin(cumulative_ang)*cumulative_s,cumulative_tx],
                           [np.sin(cumulative_ang)*cumulative_s,np.cos(cumulative_ang)*cumulative_s,cumulative_ty]])

            # Affine wrapping of current frame to the first frame orientation
            frame_stabilized = cv2.warpAffine(frame, cumulative_m, (w, h))

            # # Fix border artifacts
            # frame_stabilized = fixBorder(frame_stabilized)

            output.write(frame_stabilized)

            # Move to next frame
            prev_frame_gray = cur_gray_frame

        else:
            break
    vid.release()
    output.release()
    cv2.destroyAllWindows()





def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=50)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def Video_Stabilization(input_video_name):

    # Read input video
    cap = cv2.VideoCapture(input_video_name)

    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Set up output video
    out = cv2.VideoWriter('video_stabilization_out.avi', fourcc, fps, (w, h))

    # Read first frame
    _, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

            # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less

        # Extract traslation
        dx = m[0][0, 2]
        dy = m[0][1, 2]

        # Extract rotation angle
        da = np.arctan2(m[0][1, 0], m[0][0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])
        # frame_out = cv2.hconcat([frame_stabilized])
        frame_out = frame_stabilized

        # If the image is too big, resize it.

        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

        cv2.imshow("Stabilized", frame_out)
        cv2.waitKey(10)
        out.write(frame_out)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()
    print('finish video stabilization')
    return 'video_stabilization_out.avi'





