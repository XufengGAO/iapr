from typing import List
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from viz_utils import plotMultipleImages

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
