import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy
from scipy import signal
from scipy.interpolate import griddata
from harris_corner_detector import our_harris_corner_detector, create_corner_plots
import os


# FILL IN YOUR ID
ID1 = '307885152'
ID2 = '316524701'


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


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


def build_pyramid(image: np.ndarray, num_levels: int):
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    """INSERT YOUR CODE HERE."""
    PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                           [4, 16, 24, 16, 4],
                                           [6, 24, 36, 24, 6],
                                           [4, 16, 24, 16, 4],
                                           [1, 4, 6, 4, 1]])
    for i in range(num_levels):
        prev_lvl = pyramid[i]
        prev_lvl_fil = signal.convolve2d(prev_lvl,PYRAMID_FILTER, mode='same',boundary='symm')
        dec_lvl = prev_lvl_fil[0::2,0::2]
        pyramid.append(dec_lvl)
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int):
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]])
    Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

    I1 = I1.astype(int)
    I2 = I2.astype(int)
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same', boundary='symm')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
    It = I2 - I1
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    half_win = window_size // 2
    for i in range(half_win, I1.shape[0] - half_win):
        for j in range(half_win, I1.shape[1] - half_win):
            Ix_neigh = Ix[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
            Ix_neigh_flat = Ix_neigh.flatten()
            Iy_neigh = Iy[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
            Iy_neigh_flat = Iy_neigh.flatten()
            It_neigh = It[i - half_win:i + half_win + 1, j - half_win:j + half_win + 1]
            It_neigh_flat = It_neigh.flatten()
            b = -It_neigh_flat

            A = np.array([Ix_neigh_flat, Iy_neigh_flat]).T

            if np.linalg.matrix_rank(np.matmul(A.T, A)) < 2:
                d = np.array([0, 0])
            else:
                d = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
            du[i, j] = d[0]
            dv[i, j] = d[1]
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()
    """INSERT YOUR CODE HERE.
    Replace image_warp with something else.
    """
    h,w = image_warp.shape
    norm_factor_u = w / u.shape[1]
    norm_factor_v = h / v.shape[0]
    u = cv2.resize(u, (w,h)) * norm_factor_u
    v = cv2.resize(v, (w,h)) * norm_factor_v

    xx, yy = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    u_flat = u.flatten()
    v_flat = v.flatten()
    org_points = np.array([xx_flat,yy_flat]).T
    # new_points = np.array([xx_flat+u_flat,yy_flat+v_flat]).T
    new_points = np.array([yy_flat + v_flat,xx_flat + u_flat]).T
    img_flat = image.flatten()

    image_warp = griddata(org_points,img_flat,new_points,method='linear',fill_value=np.nan,rescale=False)

    # image_warp = scipy.ndimage.map_coordinates(image,new_points.T,output=None)
    image_warp[np.isnan(image_warp)]=img_flat[np.isnan(image_warp)]

    image_warp = image_warp.reshape(image.shape)
    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int): #-> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    """INSERT YOUR CODE HERE.
        Replace image_warp with something else.
        """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)
    """INSERT YOUR CODE HERE.
       Replace u and v with their true value."""

    for i in range(len(pyramid_I2)-1, -1, -1):
        warp_I2 = warp_image(pyramid_I2[i], u, v)
        for j in range(max_iter):
            # u, v = lucas_kanade_step(pyramid_I1[i], warp_I2, window_size)
            # warp_I2 = warp_image(warp_I2, u, v)
            du,dv = lucas_kanade_step(pyramid_I1[i], warp_I2, window_size)
            u = u + du
            v = v + dv
            warp_I2 = warp_image(pyramid_I2[i], u, v)
        if i > 0:
            h,w = pyramid_I1[i-1].shape
            norm_factor_u = w / u.shape[1]
            norm_factor_v = h/ v.shape[0]
            u = cv2.resize(u, (w,h)) * norm_factor_u
            v = cv2.resize(v, (w,h)) * norm_factor_v

    return u, v


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int): #-> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    """INSERT YOUR CODE HERE."""
    #####
    debug_dir = 'LK_stab_frames'
    os.makedirs(debug_dir, exist_ok=True)
    #####
    half_win = window_size // 2
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
                             (params['width'], params['height']), isColor=False)

    ret, frame = vid.read()
    first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h_factor = int(np.ceil(first_frame_gray.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame_gray.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))

    output.write(first_frame_gray)
    first_frame_gray_resize = cv2.resize(first_frame_gray, IMAGE_SIZE)

    u = np.zeros(first_frame_gray_resize.shape)
    v = np.zeros(first_frame_gray_resize.shape)

    prev_frame_gray = first_frame_gray_resize
    prev_u = u
    prev_v = v
    h,w = u.shape
    frame_amount = params['frame_count']

    for i in tqdm(range(1, frame_amount)):
        ret, frame = vid.read()
        if ret:
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cur_gray_frame_resize = cv2.resize(cur_gray_frame, IMAGE_SIZE)

            u, v = lucas_kanade_optical_flow(prev_frame_gray,cur_gray_frame_resize,window_size,max_iter,num_levels)
            u_avg = np.average(u[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            v_avg = np.average(v[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            frame_u = prev_u + u_avg * np.ones(u.shape)
            frame_v = prev_v + v_avg * np.ones(v.shape)
            stabilized_frame = warp_image(cur_gray_frame,frame_u,frame_v)
            stabilized_frame = stabilized_frame.astype(np.uint8)

            output.write(stabilized_frame)

            prev_frame_gray = cur_gray_frame_resize
            prev_u = frame_u
            prev_v = frame_v
        else:
            break
    vid.release()
    output.release()
    cv2.destroyAllWindows()

    pass


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int): #-> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]])
    Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

    I1 = I1.astype(int)
    I2 = I2.astype(int)

    # lucas_kanade_step
    thresh = (window_size * 10)**2

    if I1.shape[0]*I1.shape[1]<thresh:
        du,dv = lucas_kanade_step(I1,I2,window_size)
    else:
        Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same', boundary='symm')
        Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
        It = I2 - I1
        half_win = window_size // 2

        harris_thresh = 100
        K = 0.05

        output = our_harris_corner_detector(I2, K, harris_thresh)
        # create_corner_plots(I2, output, I2,
        #                     output, to_save=False)
        corners = np.array(np.where(output == 1)).T

        for i in range(corners.shape[0]):
            if corners[i,0] in range(half_win,I1.shape[0]-half_win) and corners[i,1] in range(half_win,I1.shape[1]-half_win):
                cur_h = corners[i,0]
                cur_w = corners[i,1]

                Ix_neigh = Ix[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                Ix_neigh_flat = Ix_neigh.flatten()
                Iy_neigh = Iy[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                Iy_neigh_flat = Iy_neigh.flatten()
                It_neigh = It[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                It_neigh_flat = It_neigh.flatten()
                b = -It_neigh_flat

                A = np.array([Ix_neigh_flat, Iy_neigh_flat]).T

                if np.linalg.matrix_rank(np.matmul(A.T, A)) < 2:
                    d = np.array([0, 0])
                else:
                    d = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
                du[cur_h, cur_w] = d[0]
                dv[cur_h, cur_w] = d[1]

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int): #-> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyramid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyramid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyramid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    for i in range(len(pyramid_I2)-1, -1, -1):
        warp_I2 = warp_image(pyramid_I2[i], u, v)
        for j in range(max_iter):
            du,dv = faster_lucas_kanade_step(pyramid_I1[i], warp_I2, window_size)
            u = u + du
            v = v + dv
            warp_I2 = warp_image(pyramid_I2[i], u, v)
        if i > 0:
            h,w = pyramid_I1[i-1].shape
            norm_factor_u = w / u.shape[1]
            norm_factor_v = h/ v.shape[0]
            u = cv2.resize(u, (w,h)) * norm_factor_u
            v = cv2.resize(v, (w,h)) * norm_factor_v
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int): #-> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""

    half_win = window_size // 2
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
                             (params['width'], params['height']), isColor=False)

    ret, frame = vid.read()
    first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h_factor = int(np.ceil(first_frame_gray.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame_gray.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))

    output.write(first_frame_gray)
    first_frame_gray_resize = cv2.resize(first_frame_gray, IMAGE_SIZE)

    u = np.zeros(first_frame_gray_resize.shape)
    v = np.zeros(first_frame_gray_resize.shape)

    prev_frame_gray = first_frame_gray_resize
    prev_u = u
    prev_v = v
    h, w = u.shape

    frame_amount = params['frame_count']
    for i in tqdm(range(1, frame_amount)):
        ret, frame = vid.read()
        if ret:
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cur_gray_frame_resize = cv2.resize(cur_gray_frame, IMAGE_SIZE)

            u, v = faster_lucas_kanade_optical_flow(prev_frame_gray, cur_gray_frame_resize, window_size, max_iter, num_levels)
            u_avg = np.average(u[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            v_avg = np.average(v[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            frame_u = prev_u + u_avg * np.ones(u.shape)
            frame_v = prev_v + v_avg * np.ones(v.shape)
            stabilized_frame = warp_image(cur_gray_frame, frame_u, frame_v)
            stabilized_frame = stabilized_frame.astype(np.uint8)

            output.write(stabilized_frame)

            prev_frame_gray = cur_gray_frame_resize
            prev_u = frame_u
            prev_v = frame_v
        else:
            break
    vid.release()
    output.release()
    cv2.destroyAllWindows()

    pass


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30): #-> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    half_win = window_size // 2
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
                             (params['width'], params['height']), isColor=False)

    ret, frame = vid.read()
    first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h_factor = int(np.ceil(first_frame_gray.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame_gray.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))

    output.write(first_frame_gray)

    first_frame_gray_resize = cv2.resize(first_frame_gray, IMAGE_SIZE)

    u = np.zeros(first_frame_gray_resize.shape)
    v = np.zeros(first_frame_gray_resize.shape)

    prev_frame_gray = first_frame_gray_resize
    prev_u = u
    prev_v = v
    h, w = u.shape

    frame_amount = params['frame_count']
    # frame_amount = 30
    for i in tqdm(range(1, frame_amount)):
        ret, frame = vid.read()
        if ret:
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cur_gray_frame_resize = cv2.resize(cur_gray_frame, IMAGE_SIZE)

            u, v = faster_lucas_kanade_optical_flow(prev_frame_gray, cur_gray_frame_resize, window_size, max_iter, num_levels)
            u_avg = np.average(u[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            v_avg = np.average(v[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            frame_u = prev_u + u_avg * np.ones(u.shape)
            frame_v = prev_v + v_avg * np.ones(v.shape)
            stabilized_frame = warp_image(cur_gray_frame, frame_u, frame_v)
            stabilized_frame = stabilized_frame.astype(np.uint8)

            stabilized_frame = stabilized_frame[start_rows:-end_rows, start_cols:-end_cols]
            org_h, org_w = cur_gray_frame.shape
            output.write(cv2.resize(stabilized_frame, (org_w,org_h)))
            prev_frame_gray = cur_gray_frame_resize
            prev_u = frame_u
            prev_v = frame_v
        else:
            break
    vid.release()
    output.release()
    cv2.destroyAllWindows()

    pass



def stabilization_step(I1: np.ndarray,
                    I2: np.ndarray,
                    window_size: int):
    """Faster implementation of a single Lucas-Kanade Step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """INSERT YOUR CODE HERE.
    Calculate du and dv correctly.
    """
    X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]])
    Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

    I1 = I1.astype(np.uint8)
    I2 = I2.astype(np.uint8)

    # lucas_kanade_step
    minDistance = 5
    thresh = max((minDistance)**3, minDistance*100)

    if I1.shape[0]*I1.shape[1]<thresh:
        xx, yy = np.meshgrid(range(I1.shape[1]), range(I1.shape[0]))
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        coords = np.array([xx_flat, yy_flat]).T
        du,dv = lucas_kanade_step(I1,I2,window_size)
    else:
        Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same', boundary='symm')
        Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same', boundary='symm')
        It = I2 - I1
        half_win = window_size // 2

        # harris_thresh = 100
        # K = 0.05
        # output = our_harris_corner_detector(I2, K, harris_thresh)
        # create_corner_plots(I2, output, I2,
        #                     output, to_save=False)
        # corners = np.array(np.where(output == 1)).T

        output = cv2.goodFeaturesToTrack(I2,
                                         maxCorners = 200,
                                         qualityLevel=0.01,
                                         minDistance=minDistance,
                                         blockSize=3)

        corners = np.squeeze(output.astype(int)).T
        coords = corners.copy()
        # plt.imshow(I2, cmap='gray')
        # plt.plot(corners[0], corners[1], 'ro')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()


        for i in range(corners.shape[0]):
            if corners[i,0] in range(half_win,I1.shape[0]-half_win) and corners[i,1] in range(half_win,I1.shape[1]-half_win):
                cur_h = corners[i,0]
                cur_w = corners[i,1]

                Ix_neigh = Ix[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                Ix_neigh_flat = Ix_neigh.flatten()
                Iy_neigh = Iy[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                Iy_neigh_flat = Iy_neigh.flatten()
                It_neigh = It[cur_h - half_win:cur_h + half_win + 1, cur_w - half_win:cur_w + half_win + 1]
                It_neigh_flat = It_neigh.flatten()
                b = -It_neigh_flat

                A = np.array([Ix_neigh_flat, Iy_neigh_flat]).T

                if np.linalg.matrix_rank(np.matmul(A.T, A)) < 2:
                    d = np.array([0, 0])
                else:
                    d = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
                du[cur_h, cur_w] = d[0]
                dv[cur_h, cur_w] = d[1]

    return du, dv, coords

def faster_lucas_kanade_optical_flow_new(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int): #-> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyramid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyramid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyramid_I2[-1].shape)  # create v in the size of smallest image
    """INSERT YOUR CODE HERE.
    Replace u and v with their true value."""
    for i in range(len(pyramid_I2)-1, -1, -1):
        warp_I2 = warp_image(pyramid_I2[i], u, v)
        for j in range(max_iter):
            du, dv, coords = stabilization_step(pyramid_I1[i], warp_I2, window_size)
            u = u + du
            v = v + dv
            warp_I2 = warp_image(pyramid_I2[i], u, v)
        if i > 0:
            h,w = pyramid_I1[i-1].shape
            norm_factor_u = w / u.shape[1]
            norm_factor_v = h/ v.shape[0]
            u = cv2.resize(u, (w,h)) * norm_factor_u
            v = cv2.resize(v, (w,h)) * norm_factor_v
    return u, v

def stabilization_old(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30):
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    half_win = window_size // 2
    vid = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
                             (params['width'], params['height']), isColor=False)

    ret, frame = vid.read()
    first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    h_factor = int(np.ceil(first_frame_gray.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(first_frame_gray.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))

    output.write(first_frame_gray)

    first_frame_gray_resize = cv2.resize(first_frame_gray, IMAGE_SIZE)

    u = np.zeros(first_frame_gray_resize.shape)
    v = np.zeros(first_frame_gray_resize.shape)

    prev_frame_gray = first_frame_gray_resize
    prev_u = u
    prev_v = v
    h, w = u.shape

    frame_amount = params['frame_count']
    # frame_amount = 30
    for i in tqdm(range(1, frame_amount)):
        ret, frame = vid.read()
        if ret:
            cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            cur_gray_frame_resize = cv2.resize(cur_gray_frame, IMAGE_SIZE)

            u, v = faster_lucas_kanade_optical_flow_new(prev_frame_gray, cur_gray_frame_resize, window_size, max_iter, num_levels)
            # u, v = stabilization_step(prev_frame_gray, cur_gray_frame_resize, window_size)
            u_avg = np.average(u[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            v_avg = np.average(v[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
            frame_u = prev_u + u_avg * np.ones(u.shape)
            frame_v = prev_v + v_avg * np.ones(v.shape)

            stabilized_frame = warp_image(cur_gray_frame, frame_u, frame_v)
            stabilized_frame = stabilized_frame.astype(np.uint8)

            stabilized_frame = stabilized_frame[start_rows:-end_rows, start_cols:-end_cols]
            org_h, org_w = cur_gray_frame.shape
            output.write(cv2.resize(stabilized_frame, (org_w,org_h)))
            prev_frame_gray = cur_gray_frame_resize
            prev_u = frame_u
            prev_v = frame_v
        else:
            break
    vid.release()
    output.release()
    cv2.destroyAllWindows()


# def stabilization(
#         input_video_path: str, output_video_path: str, window_size: int,
#         max_iter: int, num_levels: int, start_rows: int = 10,
#         start_cols: int = 2, end_rows: int = 30, end_cols: int = 30):
#     """Calculate LK Optical Flow to stabilize the video and save it to file.
#
#     Args:
#         input_video_path: str. path to input video.
#         output_video_path: str. path to output stabilized video.
#         window_size: int. The window is of shape window_size X window_size.
#         max_iter: int. Maximal number of LK-steps for each level of the pyramid.
#         num_levels: int. Number of pyramid levels.
#         start_rows: int. The number of lines to cut from top.
#         end_rows: int. The number of lines to cut from bottom.
#         start_cols: int. The number of columns to cut from left.
#         end_cols: int. The number of columns to cut from right.
#
#     Returns:
#         None.
#     """
#     """INSERT YOUR CODE HERE."""
#     half_win = window_size // 2
#     vid = cv2.VideoCapture(input_video_path)
#     params = get_video_parameters(vid)
#
#     # Define the codec for output video
#     # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     ##########
#     # Get width and height of video stream
#     w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # Get frames per second (fps)
#     fps = vid.get(cv2.CAP_PROP_FPS)
#
#     # Define the codec for output video
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#
#     # Set up output video
#     output = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
#     ##########
#     # output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
#     #                          (params['width'], params['height']), isColor=False)
#
#     ret, frame = vid.read()
#     first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#
#     h_factor = int(np.ceil(first_frame_gray.shape[0] / (2 ** (num_levels - 1 + 1))))
#     w_factor = int(np.ceil(first_frame_gray.shape[1] / (2 ** (num_levels - 1 + 1))))
#     IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
#                   h_factor * (2 ** (num_levels - 1 + 1)))
#
#     # output.write(first_frame_gray)
#
#     first_frame_gray_resize = cv2.resize(first_frame_gray, IMAGE_SIZE)
#
#     # u = np.zeros(first_frame_gray_resize.shape)
#     # v = np.zeros(first_frame_gray_resize.shape)
#
#     prev_frame_gray = first_frame_gray_resize
#     # prev_u = u
#     # prev_v = v
#     h, w = [params['height'],params['width']]
#
#
#
#     frame_amount = params['frame_count']
#     # frame_amount = 30
#
#     # Pre-define transformation-store array
#     transforms = np.zeros((frame_amount-1, 3), np.float32)
#
#     for i in tqdm(range(frame_amount-1)):
#         ret, frame = vid.read()
#         if ret:
#             cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#             cur_gray_frame_resize = cv2.resize(cur_gray_frame, IMAGE_SIZE)
#             prev_pnts, cur_pnts = sift_func(prev_frame_gray,cur_gray_frame_resize)
#             m = cv2.estimateAffinePartial2D(prev_pnts, cur_pnts)  # will only work with OpenCV-3 or less
#
#             # Extract traslation
#             dx = m[0][0, 2]
#             dy = m[0][1, 2]
#
#             # Extract rotation angle
#             da = np.arctan2(m[0][1, 0], m[0][0, 0])
#
#             # Store transformation
#             transforms[i] = [dx, dy, da]
#
#             # Move to next frame
#             prev_frame_gray = cur_gray_frame
#
#             # print("Frame: " + str(i) + "/" + str(frame_amount) + " -  Tracked points : " + str(len(prev_pnts)))
#         else:
#             break
#     # Compute trajectory using cumulative sum of transformations
#     trajectory = np.cumsum(transforms, axis=0)
#
#     # Create variable to store smoothed trajectory
#     smoothed_trajectory = smooth(trajectory)
#
#     # Calculate difference in smoothed_trajectory and trajectory
#     difference = smoothed_trajectory - trajectory
#
#     # Calculate newer transformation array
#     transforms_smooth = transforms + difference
#
#     # Reset stream to first frame
#     vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     # Write n_frames-1 transformed frames
#     for i in range(frame_amount - 2):
#         # Read next frame
#         success, frame = vid.read()
#         if not success:
#             break
#
#         # Extract transformations from the new transformation array
#         dx = transforms_smooth[i, 0]
#         dy = transforms_smooth[i, 1]
#         da = transforms_smooth[i, 2]
#
#         # Reconstruct transformation matrix accordingly to new values
#         m = np.zeros((2, 3), np.float32)
#         m[0, 0] = np.cos(da)
#         m[0, 1] = -np.sin(da)
#         m[1, 0] = np.sin(da)
#         m[1, 1] = np.cos(da)
#         m[0, 2] = dx
#         m[1, 2] = dy
#
#         # Apply affine wrapping to the given frame
#         frame_stabilized = cv2.warpAffine(frame, m, (w, h))
#
#         # Fix border artifacts
#         frame_stabilized = fixBorder(frame_stabilized)
#
#         # Write the frame to the file
#         # frame_out = cv2.hconcat([frame, frame_stabilized])
#         # frame_out = cv2.hconcat([frame_stabilized])
#         frame_out = frame_stabilized
#
#         # If the image is too big, resize it.
#
#         if (frame_out.shape[1] > 1920):
#             frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2))
#
#         # cv2.imshow("Stabilized", frame_out)
#         # cv2.waitKey(10)
#         output.write(frame_out)
#
#
#             # u, v = faster_lucas_kanade_optical_flow_new(prev_frame_gray, cur_gray_frame_resize, window_size, max_iter, num_levels)
#             # # u, v = stabilization_step(prev_frame_gray, cur_gray_frame_resize, window_size)
#             # u_avg = np.average(u[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
#             # v_avg = np.average(v[half_win: (h - half_win + 1), half_win:(w - half_win + 1)])
#             # frame_u = prev_u + u_avg * np.ones(u.shape)
#             # frame_v = prev_v + v_avg * np.ones(v.shape)
#             #
#             # stabilized_frame = warp_image(cur_gray_frame, frame_u, frame_v)
#             # stabilized_frame = stabilized_frame.astype(np.uint8)
#             #
#             # stabilized_frame = stabilized_frame[start_rows:-end_rows, start_cols:-end_cols]
#             # org_h, org_w = cur_gray_frame.shape
#             # output.write(cv2.resize(stabilized_frame, (org_w,org_h)))
#             # prev_frame_gray = cur_gray_frame_resize
#             # prev_u = frame_u
#             # prev_v = frame_v
#     vid.release()
#     output.release()
#     cv2.destroyAllWindows()


# def stabilization(
#         input_video_path: str, output_video_path: str, window_size: int,
#         max_iter: int, num_levels: int, start_rows: int = 10,
#         start_cols: int = 2, end_rows: int = 30, end_cols: int = 30):
#     """Calculate LK Optical Flow to stabilize the video and save it to file.
#
#     Args:
#         input_video_path: str. path to input video.
#         output_video_path: str. path to output stabilized video.
#         window_size: int. The window is of shape window_size X window_size.
#         max_iter: int. Maximal number of LK-steps for each level of the pyramid.
#         num_levels: int. Number of pyramid levels.
#         start_rows: int. The number of lines to cut from top.
#         end_rows: int. The number of lines to cut from bottom.
#         start_cols: int. The number of columns to cut from left.
#         end_cols: int. The number of columns to cut from right.
#
#     Returns:
#         None.
#     """
#     """INSERT YOUR CODE HERE."""
#     # half_win = window_size // 2
#     vid = cv2.VideoCapture(input_video_path)
#     params = get_video_parameters(vid)
#
#     # Define the codec for output video
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     # Set up output video
#     output = cv2.VideoWriter(output_video_path, fourcc, params['fps'], (params['width'], params['height']))
#     # output = cv2.VideoWriter(output_video_path, fourcc, params['fps'],
#     #                          (params['width'], params['height']), isColor=False)
#
#     ret, frame = vid.read()
#     first_frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#
#     prev_frame_gray = first_frame_gray
#     h, w = [params['height'],params['width']]
#
#     frame_amount = params['frame_count']
#     # frame_amount = 30
#
#     # Pre-define transformation-store array
#     transforms = np.zeros((frame_amount-1, 3), np.float32)
#
#     for i in tqdm(range(frame_amount-1)):
#         ret, frame = vid.read()
#         if ret:
#             cur_gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#             prev_pnts, cur_pnts = sift_func(prev_frame_gray,cur_gray_frame)
#             m = cv2.estimateAffinePartial2D(prev_pnts, cur_pnts)  # will only work with OpenCV-3 or less
#
#             # Extract traslation
#             dx = m[0][0, 2]
#             dy = m[0][1, 2]
#
#             # Extract rotation angle
#             da = np.arctan2(m[0][1, 0], m[0][0, 0])
#
#             # Store transformation
#             transforms[i] = [dx, dy, da]
#
#             # Move to next frame
#             prev_frame_gray = cur_gray_frame
#
#             # print("Frame: " + str(i) + "/" + str(frame_amount) + " -  Tracked points : " + str(len(prev_pnts)))
#         else:
#             break
#     # Compute trajectory using cumulative sum of transformations
#     trajectory = np.cumsum(transforms, axis=0)
#
#     # Create variable to store smoothed trajectory
#     smoothed_trajectory = smooth(trajectory)
#
#     # Calculate difference in smoothed_trajectory and trajectory
#     difference = smoothed_trajectory - trajectory
#
#     # Calculate newer transformation array
#     transforms_smooth = transforms + difference
#
#     # Reset stream to first frame
#     vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     # success, frame = vid.read()
#     # output.write(frame)
#
#     # Write n_frames-1 transformed frames
#     for i in range(frame_amount - 1):
#         # Read next frame
#         success, frame = vid.read()
#         if not success:
#             break
#
#         # Extract transformations from the new transformation array
#         dx = transforms_smooth[i, 0]
#         dy = transforms_smooth[i, 1]
#         da = transforms_smooth[i, 2]
#
#         # Reconstruct transformation matrix accordingly to new values
#         m = np.zeros((2, 3), np.float32)
#         m[0, 0] = np.cos(da)
#         m[0, 1] = -np.sin(da)
#         m[1, 0] = np.sin(da)
#         m[1, 1] = np.cos(da)
#         m[0, 2] = dx
#         m[1, 2] = dy
#
#         # Apply affine wrapping to the given frame
#         frame_stabilized = cv2.warpAffine(frame, m, (w, h))
#
#         # Fix border artifacts
#         frame_stabilized = fixBorder(frame_stabilized)
#
#         # Write the frame to the file
#         frame_out = frame_stabilized
#
#         # If the image is too big, resize it.
#
#         if (frame_out.shape[1] > 1920):
#             frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2))
#
#         # cv2.imshow("Stabilized", frame_out)
#         # cv2.waitKey(10)
#         output.write(frame_out)
#     vid.release()
#     output.release()
#     cv2.destroyAllWindows()

def stabilization(input_video_path: str, output_video_path: str):
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


    print('hi')



#########################################################
# Background Subtraction
def BG_subtraction(input_video_path: str, output_bg_extact_path: str, output_bg_bin_path: str):
    #####
    # Adjustable params:
    history = 500
    varThreshold = 50

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

    fgbg = cv2.createBackgroundSubtractorMOG2(history = history, varThreshold = varThreshold)

    for i in tqdm(range(frame_amount-1)):
        ret, frame = vid.read()
        if ret:
            # if i>0:
            #     fgmask = fgbg.apply(frame, fgmask, 0.005)
            # else:
            #     fgmask = fgbg.apply(frame)
            fgmask = fgbg.apply(frame)
            # plt.imshow(fgmask,cmap='gray')
            # plt.show()

            out_bin.write(fgmask)

            # cv2.imshow('frame',fgmask)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # print('hi')

    vid.release()
    out_bin.release()
    cv2.destroyAllWindows()




