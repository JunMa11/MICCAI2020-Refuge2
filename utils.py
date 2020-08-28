from skimage.morphology import remove_small_objects
import numpy as np
from skimage.measure import *
import cv2
import matplotlib.pyplot as plt

def remove_small_areas(img, min_area):
    img = img.astype(np.bool)
    img = remove_small_objects(img, min_area, connectivity=1)
    img = img.astype(np.uint8)
    return img

def keep_large_area(img, top_n_large):
    post_img = np.zeros(img.shape)
    img = img.astype(np.uint8)
    connect_regions = label(img, connectivity=1, background=0)
    props = regionprops(connect_regions)
    regions_area = []
    if len(props) > top_n_large:
        for n in range(len(props)):
            regions_area.append(props[n].area)
        index = np.argsort(np.array(regions_area))
        index = np.flip(index)
        for i in range(top_n_large):
            index_one = index[i]
            filled_value = props[index_one].label
            post_img[connect_regions == filled_value] = 1
    else:
        post_img = img
    return post_img

def fit_Ellipse(mask):
    Ellipse_mask = np.zeros(mask.shape)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0 and len(contours[0]) > 5:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(Ellipse_mask, ellipse, 1, -1)
    else:
        Ellipse_mask = mask
    return Ellipse_mask

def crop_mask_expand(mask, expand_Percentage):
    mask_copy = np.copy(mask)
    img_shape = mask_copy.shape
    x_, y_ = mask_copy.nonzero()
    x1 = x_.min()
    x2 = x_.max()
    y1 = y_.min()
    y2 = y_.max()
    expand_x = int(expand_Percentage * 100) if x2 - x1 <= 100 else int((x2 - x1) * expand_Percentage)
    expand_y = int(expand_Percentage * 100) if y2 - y1 <= 100 else int((y2 - y1) * expand_Percentage)
    x1_new = 0 if x1 - expand_x <= 0 else x1 - expand_x
    x2_new = img_shape[0] if x2 + expand_x >= img_shape[0] else x2 + expand_x
    y1_new = 0 if y1 - expand_y <= 0 else y1 - expand_y
    y2_new = img_shape[1] if y2 + expand_y >= img_shape[1] else y2 + expand_y
    return x1_new, x2_new, y1_new, y2_new

def roi_extend(img_shape, size, x1_new, x2_new, y1_new, y2_new):
    if x2_new - x1_new >= size[0]:
        x1_roi, x2_roi = x1_new, x2_new
    else:
        left_extend = int((size[0] - (x2_new - x1_new)) / 2)
        right_extend = size[0] - (x2_new - x1_new) - left_extend
        x1_roi = x1_new - left_extend
        x2_roi = x2_new + right_extend
        if x1_roi < 0:
            x1_roi = 0
            x2_roi = size[0]
        if x2_roi > img_shape[0]:
            x2_roi = img_shape[0]
            x1_roi = img_shape[0] - size[0]
    if y2_new - y1_new >= size[1]:
        y1_roi, y2_roi = y1_new, y2_new
    else:
        top_extend = int((size[1] - (y2_new - y1_new)) / 2)
        down_extend = size[1] - (y2_new - y1_new) - top_extend
        y1_roi = y1_new - top_extend
        y2_roi = y2_new + down_extend
        if y1_roi < 0:
            y1_roi = 0
            y2_roi = size[1]
        if y2_roi > img_shape[1]:
            y2_roi = img_shape[1]
            y1_roi = img_shape[1] - size[1]
    return x1_roi, x2_roi, y1_roi, y2_roi

def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img

def adjust_lr(optimizer, lr_max, epoch, all_epochs):
    if epoch < 0.1 * all_epochs:
        lr = (0.99 * lr_max * epoch) / (0.1 * all_epochs) + 0.01 * lr_max
    elif epoch < 0.6 * all_epochs:
        lr = lr_max
    elif epoch < 0.9 * all_epochs:
        lr = lr_max * 0.1
    else:
        lr = lr_max * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    img = cv2.imread('data/image/2.PNG', 0)
    img_r = rotate_bound(img, -10)
    print(img.shape)
    print(img_r.shape)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(img_r, cmap='gray')
    plt.show()