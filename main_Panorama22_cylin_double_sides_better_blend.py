import math
import os
import threading
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt

print("aaa")

def debug_print(img1, img2, img_panorama):
    print(img1.shape, img2.shape)
    plt.figure(figsize=(10, 8))

    # Create a 2 row, 1 column subplot layout and display the two original
    # images in the first subplot
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.xticks([]), plt.yticks([])

    if img_panorama is not None:
        print(img_panorama.shape)
        # Display the stitched panorama in the second subimage
        plt.subplot(2, 1, 2)
        plt.imshow(img_panorama, cmap='gray')
        plt.title('Panorama')
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()  # Automatically adjust sub-image parameters to fill the entire image area
    plt.show(block=False)  # Set block=False
    # Pause for 2 seconds
    time.sleep(2)

    # Close image window
    plt.close()



def estimate_overlap_from_left_border(img1, img2, points1, points2, good_matches):

    # Estimated overlap area width
    overlap2 = max(points2[:, 0, 0])

    # Convert to coincidence percentage
    overlap_percentage = overlap2 / img2.shape[1] * 100
    return overlap_percentage



def estimate_overlap_from_right_border(img1, img2, points1, points2, good_matches):

    overlap2 = min(points2[:, 0, 0])

    overlap_percentage = (img2.shape[1] - overlap2) / img2.shape[1] * 100
    return overlap_percentage


def getTransform(source_pts, dest_pts, method='affine'):

    # src_pts = np.float32(source_pts).reshape(-1, 1, 2)
    # dst_pts = np.float32(dest_pts).reshape(-1, 1, 2)
    src_pts = source_pts
    dst_pts = dest_pts

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    # M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, mask)


def find_color_bound(image):
    # Find the index of the column of the leftmost non-black pixel
    left_non_black_column_index = np.argmax(np.any(image > 0, axis=0))

    # Find the index of the column of the rightmost non-black pixel
    right_non_black_column_index = np.argmax(np.any(image > 0, axis=0)[::-1])

    # Calculate the x-coordinate relative to the upper left corner
    left_x_coordinate = left_non_black_column_index
    right_x_coordinate = image.shape[1] - 1 - right_non_black_column_index

    min_y = np.argmax(np.any(image > 0, axis=1))

    # Find the index of the column on the far right that is not a black pixel
    max_y = np.argmax(np.any(image > 0, axis=1)[::-1])

    # Calculate the x-coordinate relative to the upper left corner
    min_y = min_y
    max_y = image.shape[0] - 1 - max_y

    return (left_x_coordinate, right_x_coordinate, min_y, max_y)


def blend_images(img1, img2, img1_pts, img2_pts, left_to_right = True):

    w = img1.shape[1]
    h = img1.shape[0]

    (MM, mask) = getTransform(img2_pts, img1_pts)
    img_final = cv2.warpAffine(img2, MM, (img1.shape[1], img1.shape[0]))

    if left_to_right:
        img2_start, w1_, w2_, w3_ = find_color_bound(img_final)
        w1_, img1_end, w2_, w3_ = find_color_bound(img1)

        gap_len = img1_end - img2_start

        for i in range(h):
            for j in range(w):
                if img_final[i][j] == 0:
                    img_final[i][j] = img1[i][j]
                elif img1[i][j] != 0:
                    alpha = 1.0 * (j - img2_start)
                    alpha /= gap_len
                    img_final[i][j] = img_final[i][j] * alpha + img1[i][j] * (1-alpha)
    else:
        w1_, img2_end, w2_, w3_ = find_color_bound(img_final)
        img1_start, w1_, w2_, w3_ = find_color_bound(img1)

        gap_len = img2_end - img1_start

        for i in range(h):
            for j in range(w):
                if img_final[i][j] == 0:
                    img_final[i][j] = img1[i][j]
                elif img1[i][j] != 0:
                    alpha = 1.0 * (j - img1_start)
                    alpha /= gap_len
                    img_final[i][j] = img1[i][j] * alpha + img_final[i][j] * (1 - alpha)

    n_w, m_w, n_h, m_h =  find_color_bound(img_final)
    return img_final, (n_h, m_h, n_w, m_w), MM


def draw_matching(good, img1, img2, kp1, kp2, mask):
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.figure()
    plt.imshow(img3, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


MIN_MATCH_COUNT = 10
def stitch_images(img1, img2, sift, flann, estimate_overlap_func, left_to_right):
    MIN_OVERLAP_PERCENTAGE = 50  # Minimum degree of coexistence

    total_w = img1.shape[1]
    small_w = img2.shape[1]
    maskImg1 = img1.copy()

    # predict_overlap_start_width = total_w - 6 * small_w // 3
    # if predict_overlap_start_width > 0:
    #     maskImg1[:, 0:predict_overlap_start_width] = 0

    kp1, des1 = sift.detectAndCompute(maskImg1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # print("\n=================\nstart sift ")

    MAX_FEATURES = 16000

    # Work with the keys and descriptors of the first image
    if len(kp1) > MAX_FEATURES:
        # The keys are sorted by response attributes in descending order, with the first MAX_FEATURES
        indices = np.argsort([-k.response for k in kp1])[:MAX_FEATURES]
        kp1 = [kp1[i] for i in indices]
        des1 = des1[indices]
    # print(img1)
    # print(des1)

    # Work with the keys and descriptors of the second image
    if len(kp2) > MAX_FEATURES:
        indices = np.argsort([-k.response for k in kp2])[:MAX_FEATURES]
        kp2 = [kp2[i] for i in indices]
        des2 = des2[indices]
    # print(img2)
    # print(des2)
    if des2 is None:
        print("Detected new NULL image")
        return img1, img1.shape[1], -1, 2, None

    # Make sure the descriptor is of type np.float32
    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)

    # print("Descriptor 1 shape:", des1.shape, "Type:", des1.dtype)
    # print("Descriptor 2 shape:", des2.shape, "Type:", des2.dtype)
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    # print("after match")

    if len(good) > 10:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        (MM, mask) = getTransform(pts2, pts1)

        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        overlap_percentage = estimate_overlap_func(img1, img2, pts1, pts2, good)
        # print("Overlap Percentage:", overlap_percentage)

        if overlap_percentage < MIN_OVERLAP_PERCENTAGE:
            # draw_matching(good, img1, img2, kp1, kp2, mask)
            img1, colored_bounds, MM = blend_images(img1, img2, pts1, pts2, left_to_right)
            return img1, colored_bounds, overlap_percentage, len(good), MM
        else:
            # print("Skipping stitching due to too much overlap")
            pass
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    return img1, img1.shape[1], -1, len(good), None  # Or return Zero or another default image when processing fails



resize_scale = 0.2



def generate_perspective(img, ratio):
    h, w = img.shape[:2]

    # Defines the four vertices of the original rectangle and the target trapezoid
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[-0, 0], [(1 + ratio * 2) * w, 0], [(1) * w, h], [ratio * 2 * w, h]])

    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply a perspective transform
    result = cv2.warpPerspective(img, M, (int((1 + ratio * 2) * w), h))  # 设置输出图像的大小为梯形的宽度和高度
    # return result[:, int(ratio * w): int(ratio* w + w)]
    return result



def cylindricalWarpImage(img1, f, savefig=False):
    h, w = img1.shape
    # f = 198
    # f = 900
    # f = 1300
    # f = 3000
    # f = 1500
    # f = 1100
    # f = 2200
    # f = 1100
    # f = 598
    # f = 100
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # mock calibration matrix

    f = K[0, 0]

    im_h, im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h, cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0, cyl_w):
        for y_cyl in np.arange(0, cyl_h):
            theta = (x_cyl - x_c) / f
            h = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K, X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]
            cyl_mask[int(y_cyl), int(x_cyl)] = 255

    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png", bbox_inches='tight')

    # original_image = cyl.copy()
    # # 指定仿射变换的角度
    # height, width = original_image.shape[:2]
    #
    # # 确定图像的中心点在下方稍微偏移的位置
    # center = (height - 2.3 * height, width // 2)
    #
    # # 极坐标变换
    # polar_image = cv2.linearPolar(original_image, center, width * 1.2)
    #
    # debug_print(cyl, polar_image, None)
    # return (cyl, cyl_mask)
    return (generate_perspective(cyl, 0.01), cyl_mask)


def cylindrical_warp_with_focal(img, focal_length):
    """
    This functions performs cylindrical warping, but its speed is slow and deprecated.

    :param img: image contents
    :param focal_length:  focal length of images
    :return: warped image
    """
    height, width = img.shape
    cylinder_proj = np.zeros(shape=img.shape, dtype=np.uint8)

    for y in range(-int(height / 2), int(height / 2)):
        for x in range(-int(width / 2), int(width / 2)):
            cylinder_x = focal_length * math.atan(x / focal_length)
            cylinder_y = focal_length * y / math.sqrt(x ** 2 + focal_length ** 2)

            cylinder_x = round(cylinder_x + width / 2)
            cylinder_y = round(cylinder_y + height / 2)

            if (cylinder_x >= 0) and (cylinder_x < width) and (cylinder_y >= 0) and (cylinder_y < height):
                cylinder_proj[cylinder_y][cylinder_x] = img[y + int(height / 2)][x + int(width / 2)]

    # Crop black border
    # ref: http://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
    # _, thresh = cv2.threshold(cv2.cvtColor(cylinder_proj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(cylinder_proj, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])

    return cylinder_proj[y:y + h, x:x + w], None



VERTICAL_BORDER_LEN = 300

# Main process
def create_panorama_from_video(video_path, time_interval, total_percent):
    f = video_path[:-4]
    try:
        f = int(f)
    except Exception:
        f = 1100

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        return None

    frame = cv2.resize(frame, None, frame, resize_scale, resize_scale)
    standard_width = frame.shape[1]
    cv2.imwrite("img1.jpg", frame)
    stitch_list = [frame]

    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img1, mask1 = cylindricalWarpImage(img1, f) # cylindrical_warp_with_focal(img1, f) #
    img1 = cv2.copyMakeBorder(img1, VERTICAL_BORDER_LEN, VERTICAL_BORDER_LEN, 20, img1.shape[1] * 2 // 3, cv2.BORDER_CONSTANT)

    sift = cv2.SIFT_create()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    key_index = 2

    new_width = img1.shape[1]

    left_frame_arr = []
    current_frame = 2
    while True:
        success, frame = cap.read()
        if not success:
            break

        # print(current_frame, int(fps * time_interval))
        if current_frame % int(fps * time_interval) == 0:
            frame = cv2.resize(frame, None, frame, resize_scale, resize_scale)

            if new_width / standard_width < 4.1 and total_percent < 460:  # stitch to the right
                img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img2, mask2 = cylindricalWarpImage(img2, f) # cylindrical_warp_with_focal(img2, f) #
                img2 = cv2.copyMakeBorder(img2, VERTICAL_BORDER_LEN, VERTICAL_BORDER_LEN, 2, 2, cv2.BORDER_CONSTANT)
                img1copy = img1.copy()
                img1, colored_bound, overlap_percent, match_cnt, affine_matrix = stitch_images(img1, img2, sift, flann,
                                                                                               estimate_overlap_from_left_border, True)


                # if match_cnt <= 7:
                #     current_frame = int(fps * time_interval) - 4 # 匹配失败了，这一帧可能是废掉的，，需要赶紧尝试下一帧
                if match_cnt <= 10:
                    total_percent = 1460
                    continue


                if overlap_percent > 0:
                    # cv2.imshow("img1", img1)
                    # cv2.imshow("img2", img2)
                    cv2.waitKey(5000)

                    stitch_list.append(frame)
                    n_h, m_h, n_w, m_w = colored_bound
                    new_width = m_w - n_w + 20
                    # debug_print(img1copy, img2, img1)

                    total_percent += overlap_percent
                    print("width changes to ", new_width, " img width=", img2.shape[1], " total percent=", total_percent, " key_index=", key_index)
                    cv2.imwrite(f"temp/{video_path}_img2_{key_index}.jpg", frame)
                    cv2.imwrite(f"temp/{video_path}_img1_{key_index}.jpg", img1copy)

                    key_index += 1
                    if new_width + img2.shape[1] * 2 // 3 > img1.shape[1]:
                        img1 = cv2.copyMakeBorder(img1, 0, 0, 0, new_width + img2.shape[1] * 2 // 3 - img1.shape[1], cv2.BORDER_CONSTANT)  # 为img2 分配空间
            else:
                left_frame_arr.append(frame)
                # current_frame = int(fps * time_interval) * 2 // 3  # 逆向，可以多记录两倍

        current_frame += 1

    img1 = cv2.copyMakeBorder(img1, 0, 0, standard_width * 2 // 3, 0, cv2.BORDER_CONSTANT)
    # start to handle left
    good_stitch_index = 0
    for frame in left_frame_arr[::-1]:
        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img2, mask2 = cylindricalWarpImage(img2, f) # cylindrical_warp_with_focal(img2, f) #
        img2 = cv2.copyMakeBorder(img2, VERTICAL_BORDER_LEN, VERTICAL_BORDER_LEN, 2, 2, cv2.BORDER_CONSTANT)
        img1copy = img1.copy()
        img1, colored_bound, overlap_percent, match_cnt, affine_matrix = stitch_images(img1, img2, sift, flann, estimate_overlap_from_right_border, False)

        if match_cnt <= 10:
            break

        if overlap_percent > 0:
            stitch_list.append(frame)
            n_h, m_h, n_w, m_w = colored_bound
            new_width = m_w - n_w + 20
            good_stitch_index += 1
            if good_stitch_index % 4 == 1 or good_stitch_index > 7:
                debug_print(img1copy, img2, img1)
            total_percent += overlap_percent
            print("handling left images: width changes to ", new_width, "img width=", img2.shape[1], "total percent=", total_percent, " key_index=", key_index)
            cv2.imwrite(f"temp/{video_path}_img2_{key_index}.jpg", frame)
            cv2.imwrite(f"temp/{video_path}_img1_{key_index}.jpg", img1copy)
            key_index += 1
            if new_width + img2.shape[1] * 4 // 3 > img1.shape[1]:
                img1 = cv2.copyMakeBorder(img1, 0, 0, new_width + img2.shape[1] * 4 // 3 - img1.shape[1], 0,
                                          cv2.BORDER_CONSTANT)  # 为img2 分配空间



    sticher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    (status, result) = sticher.stitch(stitch_list)
    if status == cv2.STITCHER_OK:
        print("Panorama generated")
        cv2.imshow(video_path, result)
        cv2.imwrite(f"{video_path}_panorama_cv2_system.jpg", result)
        cv2.waitKey(10)

    cap.release()

    return img1


for video_path in os.listdir("."):
    if video_path.endswith(".mp4"): #    and  "1107" in video_path:
        print("handling, ", video_path)
        # Call the function
        # video_path = '1101.mp4'
        panorama = create_panorama_from_video(video_path, 0.7, 100)  # Extract a frame every 1 second for processing

        # Display the results
        if panorama is not None:
            cv2.imwrite(f"{video_path}_panorama.jpg", panorama)
            plt.imshow(panorama, cmap='gray')
            plt.show(block=False)  # set block=False
            # Pause for 2 seconds
            time.sleep(2)

            # Close the image window
            plt.close()
        else:
            print("Failed to create panorama")




# todo 1.注意拍摄视频的时候一定不要晃的太厉害。要水平
# todo 2.注意拍视频的时候，要向右手侧 旋转，
# todo  3.如果两张图片的相似的区域太多的话，如果a和b的相似度太高，可以考虑先缓存b，
#  然后看看下一张c和a的相似度，如果a和c的相似度还是大于 30%则继续缓存c，读取下面的d
#  知道下面一张比如说i的相似度低于30%了，我们就可以确认h是相似度大于30%的最后一张图片，这时候就可以拼接a和h
#   然后再从h继续后面的拼接
#   这么做的好处是 越少的拼接就会越少的颜色和亮度的不一致

#  1. 怎样融合的问题
#  2. 到了怎样的情况再进行融合的问题
#  3. 解决的第三个问题，，速度的问题，打log
#  4. todo 再去拍摄，所焦距----------
#  5. 圆柱体的 映射
#  ..\venv\Scripts\python.exe    Panorama22_cylin_double_sides.py