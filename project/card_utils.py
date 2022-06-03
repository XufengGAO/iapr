from typing import List, Dict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from viz_utils import plotMultipleImages

import math
import imutils
from scipy.spatial import distance


# TODO: check paremeters
def checkNoPlay(
    img_players: List[np.array],
    intensity_thresh: float = 0.8,
    red_thresh=0.2,
    blue_thresh=0.023,
    debug: bool = False,
    fig_title: str = None,
) -> List[bool]:
    img_players_hsv = [
        cv.cvtColor(np.array(img), cv.COLOR_BGR2HSV) for img in img_players
    ]

    img_h = [img[:, :, 0] / 255 for img in img_players_hsv]
    img_s = [img[:, :, 1] / 255 for img in img_players_hsv]

    thresh_dict = {'red': [img_h, red_thresh], 'blue': [img_s, blue_thresh]}

    playing_ = []
    for key, val in thresh_dict.items():
        img_vec = [img.ravel() for img in val[0]]
        intensity = np.array(
            [im[im > intensity_thresh].sum() / np.prod(im.shape[:2]) for im in img_vec]
        )
        playing = np.argwhere(intensity < val[1]).ravel()
        playing_.append(playing)
        if debug:
            print('## ' + key)
            print('Intensity:', intensity)
            print('Playing:', playing)
    playing_ids = np.sort(np.intersect1d(playing_[0], playing_[1]))
    player_status = [i in playing_ids for i in range(4)]
    if debug:
        print(player_status)
        plotMultipleImages(
            1,
            4,
            img_players,
            titles=['p{}: {}'.format(i, player_status[i - 1]) for i in range(1, 5)],
            cmap=['rgb', 'rgb', 'rgb', 'rgb'],
            bold_axis=[not status for status in player_status],
            fig_title=fig_title,
            figsize=(20, 6),
        )

    return player_status


#%% extract single card from table cards area

## card parameter
cardW = 57
cardH = 87
cornerXmin = 2
cornerXmax = 10.5
cornerYmin = 2.5
cornerYmax = 23

## convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
zoom = 4
cardW *= zoom
cardH *= zoom
cornerXmin = int(cornerXmin * zoom)
cornerXmax = int(cornerXmax * zoom)
cornerYmin = int(cornerYmin * zoom)
cornerYmax = int(cornerYmax * zoom)

## refernce card size
refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array(
    [[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32
)

## construct alphamask
# clean the border of the detected cards
# make that border transparent. Cards are not perfect rectangles because corners are rounded. We need to make transparent the zone between the real card and its bounding rectangle, otherwise this zone will be visible in the final generated images of the dataset
bord_size = 2  # bord_size alpha=0
alphamask = np.ones((cardH, cardW), dtype=np.uint8) * 255
cv.rectangle(alphamask, (0, 0), (cardW - 1, cardH - 1), 0, bord_size)
cv.line(alphamask, (bord_size * 3, 0), (0, bord_size * 3), 0, bord_size)
cv.line(alphamask, (cardW - bord_size * 3, 0), (cardW, bord_size * 3), 0, bord_size)
cv.line(alphamask, (0, cardH - bord_size * 3), (bord_size * 3, cardH), 0, bord_size)
cv.line(
    alphamask,
    (cardW - bord_size * 3, cardH),
    (cardW, cardH - bord_size * 3),
    0,
    bord_size,
)
# plt.figure(figsize=(10, 10))
# plt.imshow(alphamask)
# plt.show()

######################################################
#      Function to extract T-area cards              #
######################################################


def extractTableCard(
    img_part_t: np.array, use_alpha: bool = False, debug: bool = False
) -> List[np.array]:
    """extract five extract table cards from part image of table cards area

    Args:
        img_part_t (np.array): part image of table cards area
        debug (bool, optional): debug flag. Defaults to False.

    Returns:
        List[np.array]: list of five extract table cards
    """
    scale_precent = 30
    width = int(img_part_t.shape[1] * scale_precent / 100)
    height = int(img_part_t.shape[0] * scale_precent / 100)
    dim = (width, height)
    img = cv.resize(img_part_t, dim, interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    dst = cv.equalizeHist(gray)

    gaussian = cv.GaussianBlur(dst, (11, 11), 0)
    gaussian = cv.GaussianBlur(gaussian, (1, 1), 0)
    threshold1 = 175
    threshold2 = 0

    edges = cv.Canny(gaussian, threshold1, threshold2)

    h, w, ch = img.shape
    contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=lambda c: cv.arcLength(c, False), reverse=True)[:7]
    convexHulls = []

    for contour in contours:
        convexHull = cv.convexHull(contour)
        convexHulls.append(convexHull)

    convexHulls_sorted = sorted(convexHulls, key=cv.contourArea, reverse=True)[:5]

    imgwarp = None

    # We want to check that 'cnt' is the contour of a rectangular shape
    # First, determine 'box', the minimum area bounding rectangle of 'cnt'
    # Then compare area of 'cnt' and area of 'box'
    # Both areas sould be very close
    rect_contour = []

    for contour in convexHulls_sorted:
        # print(cv.contourArea(contour))
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        rect_contour.append(box)

    imgwarps_dict = {}
    for contour in rect_contour:
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        mean_x = int(np.mean(box, axis=0)[0])

        # We want transform the zone inside the contour into the reference rectangle of dimensions (cardW,cardH)
        ((xr, yr), (wr, hr), thetar) = rect
        # Determine 'Mp' the transformation that transforms 'box' into the reference rectangle
        if wr > hr:
            Mp = cv.getPerspectiveTransform(np.float32(box), refCard)
        else:
            Mp = cv.getPerspectiveTransform(np.float32(box), refCardRot)
        # Determine the warped image by applying the transformation to the image
        imgwarp = cv.warpPerspective(img, Mp, (cardW, cardH))

        # Add alpha layer
        imgwarp = cv.cvtColor(imgwarp, cv.COLOR_RGB2RGBA)

        # Shape of 'cnt' is (n,1,2), type=int with n = number of points
        # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
        cnta = contour.reshape(1, -1, 2).astype(np.float32)
        # Apply the transformation 'Mp' to the contour
        cntwarp = cv.perspectiveTransform(cnta, Mp)
        cntwarp = cntwarp.astype(np.int)

        if use_alpha:
            # We build the alpha channel so that we have transparency on the
            # external border of the card
            # First, initialize alpha channel fully transparent
            alphachannel = np.zeros(imgwarp.shape[:2], dtype=np.uint8)
            # Then fill in the contour to make opaque this zone of the card
            cv.drawContours(alphachannel, cntwarp, 0, 255, -1)

            # Apply the alphamask onto the alpha channel to clean it
            alphachannel = cv.bitwise_and(alphachannel, alphamask)

            # Add the alphachannel to the warped image
            imgwarp[:, :, 3] = alphachannel

        # Save the image to file
        # if output_fn is not None:
        #     cv.imwrite(output_fn, imgwarp)

        # imgwarps.append(imgwarp)
        imgwarps_dict[mean_x] = imgwarp

    img_table_cards = []
    # sort the dict
    # print(sorted(imgwarps_dict.keys()))
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    for key in sorted(imgwarps_dict.keys()):
        sharpen = cv.filter2D(imgwarps_dict[key], -1, sharpen_kernel)
        sharpen = cv.cvtColor(sharpen, cv.COLOR_RGBA2RGB)
        img_table_cards.append(sharpen)

    if debug:
        # image 1: contour
        img_contour = np.zeros((h, w, ch), dtype=np.uint8)
        cv.drawContours(img_contour, contours, -1, (255, 255, 255), 10)
        _, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].imshow(img_part_t)
        ax[1].imshow(img_contour)
        # image 2: 5 extracted images
        imgs_name = ['T1', 'T2', 'T3', 'T4', 'T5']
        plotMultipleImages(
            1,
            5,
            images=img_table_cards,
            titles=imgs_name,
            cmap=['rgb'] * 5,
            figsize=(20, 6),
        )

    return img_table_cards


######################################################
#      Functions to extract Player cards             #
######################################################


def getPlayerRes(
    img_players: List[np.array],
    img_ID: int,
    debug=False,
) -> List[np.array]:

    """
    function to get the player results
    """
    # results init
    chip_results = {
        "P11": 0,
        "P12": 0,
        "P21": 0,
        "P22": 0,
        "P31": 0,
        "P32": 0,
        "P41": 0,
        "P42": 0,
    }
    # rotate angles
    rotate_angles = {0: 90, 1: -180, 2: -180, 3: -90}

    total_list = []
    # iterate the player cards
    for index, img_player in enumerate(img_players):
        imgwarps_list = extractPlayerCard(img_player, rotate_angles[index], debug)
        total_list += imgwarps_list

    # just for testing
    img_title = [
        'img {} P1.1'.format(img_ID),
        'P1.2',
        'P2.1',
        'P2.2',
        'P3.1',
        'P3.2',
        'P4.1',
        'P4.2',
    ]
    """
        plotMultipleImages(1, 8, total_list, \
                               img_title,\
                               ['rgb']*8, (25,25))
    """

    return total_list


def getPerpCoord(aX, aY, bX, bY, length):
    """
    function to reorder the position of 4 corners
    """
    vX = bX - aX
    vY = bY - aY
    mag = math.sqrt(vX * vX + vY * vY)
    if mag < 0.5:  # Note: should be removed if detecting playing is enabled
        mag = 1
    vX = vX / mag
    vY = vY / mag
    # clockwise direction
    temp = vX
    vX = 0 - vY
    vY = temp

    # two direction points
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return (int(cX), int(cY)), (int(dX), int(dY))


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def getNormCoord(point_sets):

    # rename the corners
    # lt = left top, lb = left bottom, lt and lb are in the same edge of bottom card
    # rt = right top, rb = right bottom , rt and rb are in the same edge of up card
    lt = (int(point_sets[0][0]), int(point_sets[0][1]))
    lb = (int(point_sets[3][0]), int(point_sets[3][1]))
    rt = (int(point_sets[1][0]), int(point_sets[1][1]))
    rb = (int(point_sets[2][0]), int(point_sets[2][1]))

    # estimate the length of long side
    long_side_1 = math.sqrt((lt[0] - lb[0]) ** 2 + (lt[1] - lb[1]) ** 2)
    long_side_2 = math.sqrt((rt[0] - rb[0]) ** 2 + (rt[1] - rb[1]) ** 2)
    long_side = max(long_side_1, long_side_2)

    # find the symmertry points of lt, lb, rt, rb
    lt_sym, _ = getPerpCoord(lb[0], lb[1], lt[0], lt[1], long_side * 0.35)  # norm to tl
    _, lb_sym = getPerpCoord(lt[0], lt[1], lb[0], lb[1], long_side * 0.35)  # norm to tr
    _, rt_sym = getPerpCoord(rb[0], rb[1], rt[0], rt[1], long_side * 0.71)  # norm to br
    rb_sym, _ = getPerpCoord(rt[0], rt[1], rb[0], rb[1], long_side * 0.71)  # norm to bl

    # box points for further perspective transform
    bottom_box = np.zeros((4, 2))  # clockwise from lt
    bottom_box[0] = [lt[0], lt[1]]
    bottom_box[1] = [lt_sym[0], lt_sym[1]]
    bottom_box[2] = [lb_sym[0], lb_sym[1]]
    bottom_box[3] = [lb[0], lb[1]]

    up_box = np.zeros((4, 2))  # clockwise from lt
    up_box[0] = [rt_sym[0], rt_sym[1]]
    up_box[1] = [rt[0], rt[1]]
    up_box[2] = [rb[0], rb[1]]
    up_box[3] = [rb_sym[0], rb_sym[1]]

    return [bottom_box, up_box]


## refernce card size
refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array(
    [[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32
)


def extractPlayerCard(
    player: np.array, rotate_angle: int = 0, debug: bool = False
) -> List[np.array]:
    """extract overlapping player cards from part image of player cards area

    Args:
        player (np.array): part image of player cards area
        debug (bool, optional): debug flag. Defaults to False.

    Returns:
        List[np.array]: list of 2 extract player cards
    """
    # step1: Image downscale
    scale_percent = 50
    width = int(player.shape[1] * scale_percent / 100)
    height = int(player.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_player = cv.resize(player, dim, interpolation=cv.INTER_AREA)

    # step2: RGB to Gray, histogram norm, noise removal and edge detection
    gray = cv.cvtColor(resized_player, cv.COLOR_RGB2GRAY)
    dst = cv.equalizeHist(gray)
    gaussian = cv.GaussianBlur(dst, (11, 11), 0)
    gaussian = cv.GaussianBlur(gaussian, (3, 3), 0)

    # threholds for canny edge detection
    threshold1 = 185
    threshold2 = 0
    edges = cv.Canny(gaussian, threshold1, threshold2)

    # small dilate to make sure card outline is complete
    kernelDilate = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernelDilate, iterations=1)

    # rotate the image to fit the view
    edges = imutils.rotate_bound(edges, rotate_angle)
    resized_player = imutils.rotate_bound(resized_player, rotate_angle)

    # step3: find the geenral outline of cards (4-points polygon)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv.arcLength(c, False), reverse=True)[:3]

    convexHulls = []
    for contour in contours:
        convexHull = cv.convexHull(contour)
        convexHulls.append(convexHull)
    convexHulls_sorted = sorted(convexHulls, key=cv.contourArea, reverse=True)[:1]

    c = convexHulls_sorted[0]
    eps = 0.001
    while True:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, eps * peri, True)
        eps += 0.008
        if len(approx) <= 4:
            break

    if len(approx) < 4:
        return [resized_player, resized_player]

    # step4: reorder the points
    new_approx = np.zeros((4, 2))
    for ind, points in enumerate(approx):
        new_approx[ind] = points
    point_sets = order_points(
        new_approx
    )  # np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    # split the overlapping players
    boxes = getNormCoord(point_sets)
    imgwarp = None
    imgwarps_list = []
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    for box in boxes:
        Mp = cv.getPerspectiveTransform(np.float32(box), refCard)
        imgwarp = cv.warpPerspective(resized_player, Mp, (cardW, cardH))
        # Save the image to file
        sharpen = cv.filter2D(imgwarp, -1, sharpen_kernel)
        imgwarps_list.append(sharpen)

    if debug:
        # image 1: 2 extracted images
        imgs_name = ['Players', 'Bottom', 'Up']
        plotMultipleImages(
            1,
            3,
            images=[resized_player] + imgwarps_list,
            titles=imgs_name,
            cmap=['rgb'] * 3,
            figsize=(10, 6),
        )

    return imgwarps_list
