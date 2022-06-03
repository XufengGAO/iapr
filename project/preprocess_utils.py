import copy

import numpy as np
import cv2 as cv
from PIL import Image

from viz_utils import plotMultipleImages, plotBboxInPlace


def getTableOrigin(table_corners: np.ndarray) -> np.ndarray:
    origin_index = np.argmin(np.sum(table_corners, axis=1), axis=0)
    table_origin = table_corners[origin_index]
    return table_origin


def addRectOffset(rect, offset=10):
    rect_list = list(rect)
    rect_list[1] = [val + offset for val in rect_list[1]]
    rect = tuple(rect_list)
    return rect


def scaleRect(rect, scale=4.0):
    rect_list = list(rect)
    rect_list[0] = [val * scale for val in rect_list[0]]
    rect_list[1] = [val * scale for val in rect_list[1]]
    rect = tuple(rect_list)
    return rect


def transRect(rect, trans=(800, 50)):
    rect_list = list(rect)
    rect_list[0][0] += trans[0]
    rect_list[0][1] += trans[1]
    rect = tuple(rect_list)
    return rect


def regulateRect(rect):
    rect_list = list(rect)
    if rect_list[2] < -45:
        rect_list[2] += 90
    elif rect_list[2] > 45:
        rect_list[2] -= 90
    rect = tuple(rect_list)
    return rect


def cropRectFromImg(img, rect, debug=False):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (width, height))

    img_crop = cv.getRectSubPix(img_rot, size, center)

    if debug:
        print("crop size", img_crop.shape)
        print("crop image aspect ratio", img_crop.shape[0] / img_crop.shape[1])

    return img_crop, img_rot


def preprocessImg(img_origin, thresh=100):
    h, w, ch = np.array(img_origin).shape
    img_obj = np.ones((h, w, ch), dtype=np.uint8)

    img_obj_255 = img_obj * 255
    img_inverse = img_obj_255 - np.array(img_origin)
    gaussian_im = cv.GaussianBlur(np.array(img_inverse), (5, 5), 0)
    img_bgr = cv.cvtColor(gaussian_im, cv.COLOR_RGB2BGR)
    im_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    _, img_binary = cv.threshold(
        im_gray, thresh, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    # ret, img_binary = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE, (3, 3))
    img_morph = cv.morphologyEx(img_binary, cv.MORPH_ERODE, kernel, iterations=10)

    return img_obj, img_binary, img_morph


def cropTableFromPre(
    img_origin_,
    img_morph,
    dim_resize,
    kernel_sz=3,
    dist_thresh=0.2,
    edge_dilate_nr=5,
    use_convex_hull=False,
    rect_offset=80,
    rect_scale=4.0,
    rect_trans=(800, 20),
    debug=False,
):
    # sure background area
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(img_morph, cv.DIST_L2, 5)
    _, img_dist_trans_thresh = cv.threshold(
        dist_transform, dist_thresh * dist_transform.max(), 255, 0
    )
    # Finding unknown region
    img_dist_trans_thresh = np.uint8(img_dist_trans_thresh)

    thresh1, thresh2 = 175, 0
    edges = cv.Canny(img_dist_trans_thresh, thresh1, thresh2)
    edges = cv.dilate(
        edges, np.ones((kernel_sz, kernel_sz), np.uint8), iterations=edge_dilate_nr
    )

    contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # contour = sorted(contours, key=cv.contourArea,reverse=True)[-1]
    contour = sorted(contours, key=lambda c: cv.arcLength(c, False), reverse=True)[0]

    img_contour = np.zeros(dim_resize[::-1], dtype=np.uint8)
    if use_convex_hull:
        contour = cv.convexHull(contour)
    cv.drawContours(img_contour, [contour], -1, (255, 255, 255), 10)

    # find suitable rectangle
    rect = cv.minAreaRect(contour)
    rect = addRectOffset(rect, offset=rect_offset)
    rect = regulateRect(rect)

    # crop resized table
    # img_np = np.asarray(img_origin)
    # img_crop, img_rot = cropRectFromImg(img_np, rect)

    # crop in original coordinate
    img_np_origin = np.asarray(img_origin_)
    rect_origin = scaleRect(rect, scale=rect_scale)
    rect_origin = transRect(rect_origin, trans=rect_trans)
    img_crop_origin, img_rot_origin = cropRectFromImg(
        img_np_origin, rect_origin, debug=debug
    )

    imgs_debug = {}
    if debug:
        img_origin_rect = copy.deepcopy(img_np_origin)
        box = cv.boxPoints(rect_origin)  # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv.drawContours(img_origin_rect, [box], 0, (0, 0, 255), 10)
        img_origin_rect = cv.resize(
            img_origin_rect, dim_resize, interpolation=cv.INTER_AREA
        )

        imgs_debug = {
            'img_rot_origin': img_rot_origin,
            'img_dist_trans_thresh': img_dist_trans_thresh,
            'img_contour': img_contour,
            'img_origin_rect': img_origin_rect,
        }

    return img_crop_origin, rect, rect_origin, imgs_debug


def cropTable(
    file,
    kernel_sz=3,
    dist_thresh=0.2,
    edge_dilate_nr=5,
    use_convex_hull=False,
    resize_flag=True,
    resize_scale=4.0,
    crop_sz=(800, 50),
    rect_offset=40,
    debug=False,
):
    img_origin_ = Image.open(file)
    w, h = img_origin_.size
    # 800, 50 | 400, 20
    img_origin_crop = img_origin_.crop(
        (crop_sz[0], crop_sz[1], w - crop_sz[0], h - crop_sz[1])
    )
    img_origin = np.asarray(img_origin_crop)
    dim_resize = (img_origin.shape[1], img_origin.shape[0])
    if resize_flag:
        dim_resize = tuple([int(val / resize_scale) for val in dim_resize])
        img_origin = cv.resize(img_origin, dim_resize, interpolation=cv.INTER_AREA)

    _, img_binary, img_morph = preprocessImg(img_origin, thresh=100)

    imgs_preprocess = {
        'img_binary': img_binary,
        'img_morph': img_morph,
    }

    img_crop_origin, rect, rect_origin, imgs_debug = cropTableFromPre(
        img_origin_,
        img_morph,
        dim_resize,
        kernel_sz=kernel_sz,  # 3
        dist_thresh=dist_thresh,  # 0.2
        edge_dilate_nr=edge_dilate_nr,  # 5
        use_convex_hull=use_convex_hull,  # False
        rect_offset=rect_offset,
        rect_scale=resize_scale,
        rect_trans=crop_sz,
        debug=debug,
    )

    if debug:
        imgs_debug = {**imgs_preprocess, **imgs_debug}
    else:
        imgs_debug = {}

    return (
        img_crop_origin,
        rect,
        rect_origin,
        imgs_debug,
    )


PART_NAMES = ['p1', 'p2', 'p3', 'p4', 'T1-T5', 'CR-CW']


def cropImgParts(
    table_crop,
    im_names=PART_NAMES,
    fig_title=None,
    viz_parts=False,
    viz_inplace=False,
):
    # train_size = np.loadtxt(os.path.join(data_path, "train_size.txt")).astype(int)
    # w1,h1 w2,h2
    tables_crop_pil = Image.fromarray(table_crop)
    w, h = tables_crop_pil.size

    box_p1 = (w - 1101, 1000, w - 1, 2500)
    box_p2 = (2000, 0, 3200, 1000)
    box_p3 = (500, 0, 1700, 1000)
    box_p4 = (0, 1000, 1100, 2500)

    box_T = (500, 2700, 3250, h - 1)
    box_C = (1000, 1000, 2600, 2600)
    boxes = [box_p1, box_p2, box_p3, box_p4, box_T, box_C]
    im_parts = []
    for box in boxes:
        im = tables_crop_pil.crop(box)
        im_parts.append(im)

    if viz_parts:
        plotMultipleImages(
            2, 3, im_parts, im_names, cmap=['rgb'] * 6, fig_title=fig_title
        )
    if viz_inplace:
        plotBboxInPlace(tables_crop_pil, boxes, im_names, fig_title=fig_title)
    return im_parts


#%% deprecated cropping
# Pre-processing
def preprocessing(im_origin='', thresh=100):
    h, w, ch = np.array(im_origin).shape
    obj_img = np.ones((h, w, ch), dtype=np.uint8)
    obj_img_255 = obj_img * 255

    img_inverse = obj_img_255 - np.array(im_origin)
    gaussian_im = cv.GaussianBlur(np.array(img_inverse), (5, 5), 0)
    img_bgr = cv.cvtColor(gaussian_im, cv.COLOR_RGB2BGR)
    im_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    print("Using threshold:", thresh)
    ret, binary = cv.threshold(
        im_gray, thresh, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    # ret, binary = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE, (6, 6))
    im_morph = cv.morphologyEx(binary, cv.MORPH_ERODE, kernel, iterations=10)
    return obj_img, im_morph, binary


# Extraction of the maximum contour, i.e. the Table
def tableCutting(obj_img, im_morph):

    # Detection of maximum contours
    contours, hierarchy = cv.findContours(
        im_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    area = []
    for k in range(len(contours)):
        area.append(cv.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    # Visualization
    # cv.drawContours(obj_img, contours, max_idx, [255,255,255], cv.FILLED)
    # plt.figure(figsize=(12,12))
    # plt.subplot(1,2,1)
    # plt.imshow(obj_img)
    # plt.title('Longest Contour')

    # External polygon fitting, table top left coordinate extraction
    rect = cv.minAreaRect(
        contours[max_idx]
    )  # Get the (centre(x,y), (width,height), rotation angle) of the smallest outer rectangle
    box = cv.boxPoints(
        rect
    )  # Get the coordinates of the 4 vertices of the smallest outer rectangle(ref: cv2.boxPoints(rect) for OpenCV 3.x)
    # converting bounding box floating point values to int in OpenCV problems.
    # box = np.int0(box)
    # o_index = np.argmin(np.sum(box, axis=1), axis=0)
    table_corners = np.int0(box)

    # Visularization
    # cv.circle(obj_img, tuple(box[o_index]), 8, (0, 255, 255), 150)
    # cv.drawContours(obj_img, [box], 0, (255, 0, 0), 20)
    # plt.subplot(122)
    # plt.imshow(obj_img)
    # plt.title('minAreaRect')
    # plt.show()
    # return box[o_index]
    return rect, table_corners


# Crop, currently only T label.
def crop(o, im_origin, viz=True, fig_title=None):
    # train_size = np.loadtxt(os.path.join(data_path, "train_size.txt")).astype(int)
    # w1,h1,w2,h2，
    box_p1 = (o[0] + 2600, o[1] + 1000, o[0] + 3760, o[1] + 2400)
    box_p2 = (o[0] + 1750, o[1] - 50, o[0] + 3216, o[1] + 880)
    box_p3 = (o[0] + 216, o[1] - 50, o[0] + 1700, o[1] + 880)
    box_p4 = (o[0] - 50, o[1] + 1200, o[0] + 1000, o[1] + 2265)

    box_T = (o[0], o[1] + 2601, o[0] + 3765, o[1] + 3765)
    box_C = (o[0] + 660, o[1] + 865, o[0] + 2600, o[1] + 2465)
    boxes = [box_p1, box_p2, box_p3, box_p4, box_T, box_C]
    im_parts = []
    for box in boxes:
        im = im_origin.crop(box)
        im_parts.append(im)
    # train_ims_crop = []
    # train_ims_crop.append(ims)
    im_names = ['P1', 'p2', 'p3', 'p4', 'T1-T5', 'CR-CW']
    if viz:
        plotMultipleImages(
            1, 6, im_parts, im_names, cmap=['rgb'] * 6, fig_title=fig_title
        )
    return im_parts


#%% deprecated utilities
# https://stackoverflow.com/a/41075028
# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
def addContrast(img):

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    return img2


# ref: https://stackoverflow.com/questions/56017384/concave-mask-to-convex
def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return cv.cvtColor(mask, cv.COLOR_BGR2GRAY)


def dilate_mask(mask, kernel_size=10):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv.dilate(mask, kernel, iterations=1)
    return dilated


def find_contours(img, to_gray=None):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    morphed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    contours = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours
