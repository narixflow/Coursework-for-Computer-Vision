import math
import os
import threading
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt


def debug_print(img1, img2, img_panorama):
    print(img1.shape, img2.shape)
    plt.figure(figsize=(10, 8))

    # 创建一个2行，1列的子图布局，并在第一个子图中显示两张原始图像
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
        # 在第二个子图中显示拼接后的全景图
        plt.subplot(2, 1, 2)
        plt.imshow(img_panorama, cmap='gray')
        plt.title('Panorama')
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.show(block=False)  # 设置 block=False
    # 暂停2秒钟
    time.sleep(2)

    # 关闭图像窗口
    plt.close()



def estimate_overlap_from_left_border(img1, img2, points1, points2, good_matches):

    # 估计重合区域宽度
    overlap2 = max(points2[:, 0, 0])

    # 转换为重合度百分比
    overlap_percentage = overlap2 / img2.shape[1] * 100
    return overlap_percentage



def estimate_overlap_from_right_border(img1, img2, points1, points2, good_matches):

    # 估计重合区域宽度
    overlap2 = min(points2[:, 0, 0])

    # 转换为重合度百分比
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



def blend_images(img1, img2, img1_pts, img2_pts):

    w = img1.shape[1]
    h = img1.shape[0]

    (MM, mask) = getTransform(img2_pts, img1_pts)
    img_final = cv2.warpAffine(img2, MM, (img1.shape[1], img1.shape[0]))

    max_colored_w = 0
    min_colored_w = w
    for i in range(h):
        for j in range(w):
            if img_final[i][j] == 0:
                img_final[i][j] = img1[i][j]
            elif img1[i][j] != 0:
                img_final[i][j] = img_final[i][j] / 2 + img1[i][j] / 2

            if img_final[i][j] > 2 and j > max_colored_w:
                max_colored_w = j
            if img_final[i][j] > 2 and j < min_colored_w:
                min_colored_w = j


    return img_final, max_colored_w - min_colored_w + 20


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
    print('!!!')


MIN_MATCH_COUNT = 10
def stitch_images(img1, img2, sift, flann, estimate_overlap_func):
    MIN_OVERLAP_PERCENTAGE = 50  # 最小重合度

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

    # 处理第一幅图像的关键点和描述符
    if len(kp1) > MAX_FEATURES:
        # 对关键点按response属性降序排序，并取前MAX_FEATURES个
        indices = np.argsort([-k.response for k in kp1])[:MAX_FEATURES]
        kp1 = [kp1[i] for i in indices]
        des1 = des1[indices]
    # print(img1)
    # print(des1)

    # 处理第二幅图像的关键点和描述符
    if len(kp2) > MAX_FEATURES:
        # 对关键点按response属性降序排序，并取前MAX_FEATURES个
        indices = np.argsort([-k.response for k in kp2])[:MAX_FEATURES]
        kp2 = [kp2[i] for i in indices]
        des2 = des2[indices]
    # print(img2)
    # print(des2)
    if des2 is None:
        print("Detected new NULL image")
        return img1, img1.shape[1], -1, 2

    # 确保描述符是 np.float32 类型
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
        # draw_matching(good, img1, img2, kp1, kp2, mask)

        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        overlap_percentage = estimate_overlap_func(img1, img2, pts1, pts2, good)
        # print("Overlap Percentage:", overlap_percentage)

        if overlap_percentage < MIN_OVERLAP_PERCENTAGE:
            img1, max_colored_w = blend_images(img1, img2, pts1, pts2)
            return img1, max_colored_w, overlap_percentage, len(good)
        else:
            # print("Skipping stitching due to too much overlap")
            pass
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    return img1, img1.shape[1], -1, len(good)  # 或者处理失败时返回None或其他默认图像



resize_scale = 0.2

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

    return (cyl, cyl_mask)

VERTICAL_BORDER_LEN = 300

# 主流程
def create_panorama_from_video(video_path, time_interval, total_percent):
    f = video_path[:-4]
    f = int(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        return None

    frame = cv2.resize(frame, None, frame, resize_scale, resize_scale)
    standard_width = frame.shape[1]
    cv2.imwrite("img1.jpg", frame)
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img1, mask1 = cylindricalWarpImage(img1, f)
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
                img2, mask2 = cylindricalWarpImage(img2, f)
                img2 = cv2.copyMakeBorder(img2, VERTICAL_BORDER_LEN, VERTICAL_BORDER_LEN, 2, 2, cv2.BORDER_CONSTANT)
                img1copy = img1.copy()
                img1, new_width, overlap_percent, match_cnt = stitch_images(img1, img2, sift, flann, estimate_overlap_from_left_border)

                # if match_cnt <= 7:
                #     current_frame = int(fps * time_interval) - 4 # 匹配失败了，这一帧可能是废掉的，，需要赶紧尝试下一帧
                if match_cnt <= 10:
                    total_percent = 1460
                    continue


                if overlap_percent > 0:
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
        img2, mask2 = cylindricalWarpImage(img2, f)
        img2 = cv2.copyMakeBorder(img2, VERTICAL_BORDER_LEN, VERTICAL_BORDER_LEN, 2, 2, cv2.BORDER_CONSTANT)
        img1copy = img1.copy()
        img1, new_width, overlap_percent, match_cnt = stitch_images(img1, img2, sift, flann, estimate_overlap_from_right_border)

        if match_cnt <= 10:
            break

        if overlap_percent > 0:
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

    cap.release()
    return img1


for video_path in os.listdir("."):
    if video_path.endswith(".mp4"):
        print("handling, ", video_path)
        # 调用函数
        # video_path = '1101.mp4'
        panorama = create_panorama_from_video(video_path, 0.7, 100)  # 每1秒提取一帧进行处理

        # 显示结果
        if panorama is not None:
            cv2.imwrite(f"{video_path}_panorama.jpg", panorama)
            plt.imshow(panorama, cmap='gray')
            plt.show(block=False)  # 设置 block=False
            # 暂停2秒钟
            time.sleep(2)

            # 关闭图像窗口
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