import scipy.stats as st
# import wdt
# import GeodisTK
from numpy import linalg as LA
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import MCP

# def likelihood_FG_BG(frame, binary, scribbles_number, **kwargs):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # convert to hsv color space and we will use only the value channel
#     # of the binary frame as scribbles
#     v = hsv[:,:,2]
#     frame_shape = frame.shape
#     scribbles_x = np.random.randint(0, frame_shape[0], scribbles_number)
#     scribbles_y = np.random.randint(0, frame_shape[1], scribbles_number)
#     F_colors = []
#     B_colors = []
#     F_scrbls = np.zeros([frame_shape[0],frame_shape[1]])
#     B_scrbls = np.zeros([frame_shape[0],frame_shape[1]])
#
#     for i in range(scribbles_number):
#         w , l = scribbles_x[i], scribbles_y[i]
#         if (binary[w,l]):
#             F_colors.append(v[w,l])
#             F_scrbls[w,l] = 255
#         else:
#             B_colors.append(v[w,l])
#             B_scrbls[w,l] = 255
#
#     # using kernel density estimation on the scribbles, the probability value for each scribble
#     # (based on its value channel from HSV) is the sum of the gaussians
#     # (we added for every scribble as center) in different distances
#
#     x_grid = np.linspace(0,255,256)
#     if(len(B_colors)):
#         x = np.array(B_colors)
#         kde = st.gaussian_kde(x, bw_method=0.2 / x.std(ddof=1), **kwargs)
#         kde_notNorm = kde.evaluate(x_grid)
#         B_pdf = kde_notNorm / np.sum(kde_notNorm)
#     else:
#         B_pdf = np.zeros(256)
#     if(len(F_colors)):
#         x = np.array(F_colors)
#         kde = st.gaussian_kde(x, bw_method=0.2 / x.std(ddof=1), **kwargs)
#         kde_notNorm = kde.evaluate(x_grid)
#         F_pdf = kde_notNorm / np.sum(kde_notNorm)
#     else:
#         F_pdf = np.zeros(256)
#     P_b = np.array([])
#     P_f = np.array([])
#     for i in range(255, -1, -1):
#         if B_pdf[i]+F_pdf[i] == 0:
#             P_b = np.insert(P_b,0,0)
#             P_f = np.insert(P_f,0,0)
#         else:
#             P_b = np.insert(P_b,0, B_pdf[i]/(B_pdf[i]+F_pdf[i]) )
#             P_f = np.insert(P_f,0,F_pdf[i]/(B_pdf[i]+F_pdf[i]) )
#     return P_b, P_f


def likelihood_FG_BG(frame, bin_mask, num_of_scribbles, **kwargs):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # convert to hsv color space and we will use only the value channel
    # of the binary frame as scribbles
    v = hsv[:,:,2]
    frame_shape = frame.shape
    scribbles_x = np.random.randint(0, frame_shape[0], num_of_scribbles)
    scribbles_y = np.random.randint(0, frame_shape[1], num_of_scribbles)
    fg_vals = []
    bg_vals = []
    fg_coords = []
    bg_coords = []
    fg_scrbls = np.zeros([frame_shape[0],frame_shape[1]])
    bg_scrbls = np.zeros([frame_shape[0],frame_shape[1]])

    for i in range(num_of_scribbles):
        w , l = scribbles_x[i], scribbles_y[i]
        if (bin_mask[w,l]):
            fg_vals.append(v[w,l])
            fg_coords.append([w, l])
            fg_scrbls[w,l] = 255
        else:
            bg_vals.append(v[w,l])
            bg_coords.append([w, l])
            bg_scrbls[w,l] = 255

    x_grid = np.linspace(0,255,256)
    if(len(bg_vals)):
        x = np.array(bg_vals)
        kde = st.gaussian_kde(x, bw_method=0.2 / x.std(ddof=1), **kwargs)
        kde_notNorm = kde.evaluate(x_grid)
        bg_pdf = kde_notNorm / np.sum(kde_notNorm)
    else:
        bg_pdf = np.zeros(256)
    if(len(fg_scrbls)):
        x = np.array(fg_vals)
        kde = st.gaussian_kde(x, bw_method=0.2 / x.std(ddof=1), **kwargs)
        kde_notNorm = kde.evaluate(x_grid)
        fg_pdf = kde_notNorm / np.sum(kde_notNorm)
    else:
        fg_pdf = np.zeros(256)
    P_b = np.array([])
    P_f = np.array([])
    for i in range(255, -1, -1):
        if bg_pdf[i]+fg_pdf[i] == 0:
            P_b = np.insert(P_b,0,0)
            P_f = np.insert(P_f,0,0)
        else:
            P_b = np.insert(P_b,0, bg_pdf[i]/(bg_pdf[i]+fg_pdf[i]))
            P_f = np.insert(P_f,0,fg_pdf[i]/(bg_pdf[i]+fg_pdf[i]))
    return P_b, P_f, bg_coords, fg_coords


def prob_maps(frame, P_b, P_f):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    h, w = v.shape
    prob_map_bg, prob_map_fg = np.zeros([h, w]), np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            prob_map_bg[i, j] = P_b[v[i,j]]
            prob_map_fg[i, j] = P_f[v[i,j]]
    return prob_map_bg, prob_map_fg


# def gradient_maps(p_map_B,p_map_F):
#     gradient_map_B_x = cv2.Sobel(p_map_B, -1, 1, 0, ksize = 5)
#     gradient_map_F_x = cv2.Sobel(p_map_F, -1, 1, 0, ksize = 5)
#     gradient_map_B_y = cv2.Sobel(p_map_B, -1, 0, 1, ksize = 5)
#     gradient_map_F_y = cv2.Sobel(p_map_F, -1, 0, 1, ksize = 5)
#
#     gradient_map_B = np.zeros(gradient_map_B_x.shape)
#     gradient_map_F = np.zeros(gradient_map_F_x.shape)
#     for i, line in enumerate(gradient_map_B):
#         for j, value_of_pixel in enumerate(line):
#             gradient_map_B[i,j] = min((LA.norm([gradient_map_B_x[i,j],gradient_map_B_y[i,j]]))**1.4, 255)
#             gradient_map_F[i,j] = min((LA.norm([gradient_map_F_x[i,j],gradient_map_F_y[i,j]]))**1.4, 255)
#     return gradient_map_B, gradient_map_F

def gradient_maps(prob_map_bg,prob_map_fg):
    vgrad_bg = np.gradient(prob_map_bg)
    gradient_map_B = np.sqrt(vgrad_bg[0] ** 2 + vgrad_bg[1] ** 2)

    vgrad_fg = np.gradient(prob_map_fg)
    gradient_map_F = np.sqrt(vgrad_fg[0] ** 2 + vgrad_fg[1] ** 2)
    return gradient_map_B, gradient_map_F

def alpha_opacity_map(P_f,P_b,distance_map_F,distance_map_B):
    w_f = np.zeros(distance_map_F.shape)
    w_b = np.zeros(distance_map_B.shape)
    alpha = np.zeros(distance_map_B.shape)

    # we chose r=2 (best results for our video) as recommended in the matting article
    r = 2
    for i in range(P_f.shape[0]):
        for j in range(P_f.shape[1]):
            try:
                w_f[i, j] = P_f[i, j]/((distance_map_F[i, j])**r)
            except :
                w_f[i, j] = w_f[i-1, j]

            try:
                w_b[i, j] = P_b[i,j]/((distance_map_B[i,j])**r)
            except:
                w_f[i, j] = w_f[i-1, j]

            if w_f[i,j] == np.inf:
                alpha[i, j] = 1
            else:
                alpha[i,j] = (w_f[i,j]/(w_f[i,j]+w_b[i,j]))
            if alpha[i,j] > 1 :
                alpha[i, j] = 1
    return alpha

def fg_bg_weighted_with_alpha(alpha,new_background,frame):
    matting_output_frame = np.zeros(new_background.shape)
    for i in range(new_background.shape[0]):
        for j in range(new_background.shape[1]):
            try:
                matting_output_frame[i, j, 0] = alpha[i, j] * frame[i, j, 0] + (1 - alpha[i, j]) * new_background[i, j, 0]
                matting_output_frame[i, j, 1] = alpha[i, j] * frame[i, j, 1] + (1 - alpha[i, j]) * new_background[i, j, 1]
                matting_output_frame[i, j, 2] = alpha[i, j] * frame[i, j, 2] + (1 - alpha[i, j]) * new_background[i, j, 2]
            except:
                matting_output_frame[i, j] = new_background[i, j]
    matting_output_frame = matting_output_frame.astype(np.uint8)

    return matting_output_frame


def create_boundary_box(binary,frame_after_matting):
    x, y, w, h = cv2.boundingRect(binary)
    frame_after_matting = cv2.rectangle(frame_after_matting, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame_after_matting


def calc_costs_wdt(grad_map_F, grad_map_B, fg_erose, fg_dilate):
    fg_d_map = wdt.map_image_to_costs(grad_map_F, fg_erose, ~fg_dilate)
    bg_d_map = wdt.map_image_to_costs(grad_map_B, ~fg_dilate, fg_erose)

    fg_cost = wdt.get_weighted_distance_transform(fg_d_map)
    bg_cost = wdt.get_weighted_distance_transform(bg_d_map)

    return fg_cost, bg_cost

def matting_tracking(input_stab_path, input_fgbg_mask, new_bg_path):
    stabilize_cap = cv2.VideoCapture(input_stab_path)
    n_frames = int(stabilize_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(stabilize_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(stabilize_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scribbles_number = int(0.1 * w * h)
    fps = stabilize_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('video_matting_tracking_out.avi', fourcc, fps, (w, h))
    binary_cap = cv2.VideoCapture(input_fgbg_mask)
    new_background = cv2.imread(new_bg_path)
    kernel = np.array([[0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0,]]).astype(np.uint8)
    path1 = r'{}\matting_tracked_frames'.format(os.getcwd())
    os.makedirs(path1, exist_ok = True)

    # loop on all frames
    for i in range(n_frames-1):
        ret, frame = stabilize_cap.read()
        ret_bin, bin_frame = binary_cap.read()
        if i < 180:
            continue
        if not ret and ret_bin:
            break

        print('matting - frame', int(i+1))
        # plt.imshow(bin_frame,cmap='gray')
        # plt.show()
        bgfg_mask = cv2.cvtColor(bin_frame, cv2.COLOR_BGR2GRAY)
        # plt.imshow(bgfg_mask, cmap='gray')
        # plt.show()
        bgfg_mask = cv2.threshold(bgfg_mask, 40, 255, cv2.THRESH_BINARY)[1]
        # plt.imshow(bgfg_mask, cmap='gray')
        # plt.show()

        # fg_erose = cv2.erode(grayFrame,kernel, iterations=2)
        # fg_dilate = cv2.dilate(grayFrame,kernel, iterations=1)
        # scribbles_coord_list = []
        # for i in range(scribbles_number):
        #     scribbles_coord_list.append([np.random.randint(0, bgfg_mask.shape[0]),
        #                                  np.random.randint(0, bgfg_mask.shape[1])])

        P_b, P_f, bg_coords, fg_coords = likelihood_FG_BG(frame, bgfg_mask, scribbles_number)

        prob_map_bg,prob_map_fg = prob_maps(frame, P_b, P_f)

        # plt.subplot(1, 2, 1)
        # plt.imshow(prob_map_fg, cmap='gray')
        # plt.title('FG prob Map')
        # plt.subplot(1, 2, 2)
        # plt.imshow(prob_map_bg, cmap='gray')
        # plt.title('BG prob Map')
        # plt.tight_layout()
        # plt.show()

        gradient_map_B, gradient_map_F = gradient_maps(prob_map_bg,prob_map_fg)

        # gradient_map_B2, gradient_map_F2 = gradient_maps2(p_map_B, p_map_F)

        # plt.subplot(2, 2, 1)
        # plt.imshow(gradient_map_B)
        # plt.title('bg1')
        # plt.subplot(2, 2, 2)
        # plt.imshow(gradient_map_F)
        # plt.title('fg1')
        # plt.subplot(2, 2, 3)
        # plt.imshow(gradient_map_B2)
        # plt.title('bg2')
        # plt.subplot(2, 2, 4)
        # plt.imshow(gradient_map_F2)
        # plt.title('fg2')
        # plt.tight_layout()
        # plt.show()


        mcp_fg = MCP(gradient_map_F)
        fg_dist_map, _ = mcp_fg.find_costs(fg_coords)

        mcp_bg = MCP(gradient_map_B)
        bg_dist_map, _ = mcp_bg.find_costs(bg_coords)

        fg_dist_map = fg_dist_map/np.max(fg_dist_map)
        bg_dist_map = bg_dist_map / np.max(bg_dist_map)

        # plt.subplot(1, 2, 1)
        # plt.imshow(fg_dist_map)
        # plt.title('FG Distance Map')
        # plt.subplot(1, 2, 2)
        # plt.imshow(bg_dist_map)
        # plt.title('BG Distance Map')
        # plt.tight_layout()
        # plt.show()

        trimap = np.zeros(fg_dist_map.shape)
        trimap[fg_dist_map<bg_dist_map] = 1

        plt.subplot(1, 2, 1)
        plt.imshow(trimap, cmap='gray')
        plt.title('Before')

        open_close_fil_size = 9
        num_of_iter = 2

        trimap = cv2.morphologyEx(trimap, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        open_close_fil_size, open_close_fil_size)), iterations=num_of_iter)
        trimap = cv2.morphologyEx(trimap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (
        open_close_fil_size, open_close_fil_size)), iterations=num_of_iter)


        # plt.subplot(1, 2, 2)
        # plt.imshow(trimap, cmap='gray')
        # plt.title('After')
        # plt.tight_layout()
        # plt.show()

        plt.subplot(1, 2, 1)
        plt.imshow(trimap, cmap='gray')
        plt.title('Background Subtraction')
        plt.subplot(1, 2, 2)
        plt.imshow(bgfg_mask, cmap='gray')
        plt.title('Trimap')
        plt.tight_layout()
        plt.show()


        # calculate the discrete weighted geodestic distance with the shortest path from every pixel to
        # the scribble set of points with the gradient of the probability map
        # distance_map_F, distance_map_B = calc_costs_wdt(gradient_map_F, gradient_map_B, fg_erose, fg_dilate)

        # sharpening the opacity map and give it more continuous values, Refinement, finding again the distance map
        # only to the undecided region(with the outside and inside contours as the scribbles)
        alpha = alpha_opacity_map(p_map_F, p_map_B, distance_map_F, distance_map_B)

        # segmented foreground on the new background (input) by weighting them with alpha
        matting_output_frame = fg_bg_weighted_with_alpha(alpha, new_background,frame)

        # tracking the object using the binary frame
        binary = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        tracking_output_frame = create_boundary_box(binary, matting_output_frame)

        cv2.imwrite(os.path.join(path1, 'matting_tracked_frame_no_' + str(i+1) + '.png'), tracking_output_frame)
        out.write(np.uint8(tracking_output_frame))

    print("finished matting and tracking")
    stabilize_cap.release()
    binary_cap.release()
    out.release()
    cv2.destroyAllWindows()