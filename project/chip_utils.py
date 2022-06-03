import copy
from typing import Dict

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from viz_utils import plotMultipleImages


def HSV_range(chip_type):
    if chip_type == "CR":
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
    elif chip_type == "CG":
        lower1 = np.array([34, 0, 0])
        upper1 = np.array([95, 255, 188])

    elif chip_type == "CB":
        lower1 = np.array([99, 100, 90])
        upper1 = np.array([160, 255, 242])

    elif chip_type == "CW":
        lower1 = np.array([90, 130, 0])
        upper1 = np.array([115, 255, 255])

    return lower1, upper1


def RGB_chip_identify(img_chip, use_full_mask=False, full_mask=None):
    """
    function to identify position of chips
    """
    h, w, ch = img_chip.shape
    gray = cv.cvtColor(img_chip, cv.COLOR_RGB2GRAY)  # rgb to gray
    gray = cv.GaussianBlur(gray, (21, 21), 1)
    ret, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )  # Otsu's thresholding after Gaussian filtering

    if use_full_mask != False:
        chip_fake_binary = np.zeros(binary.shape, binary.dtype)
        chip_fake_binary[np.where(full_mask == 255)] = binary[
            np.where(full_mask == 255)
        ]
    else:
        # print("false")
        chip_fake_binary = binary

    kernelClose = np.ones((11, 11), np.uint8)
    opening = cv.morphologyEx(chip_fake_binary, cv.MORPH_OPEN, (5, 5), iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernelClose, iterations=1)

    dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(
        dist_transform, 0.5 * dist_transform.max(), 255, cv.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    # plotMultipleImages(1, 3, [img_chip, binary, sure_fg], ['img_chip', 'binary {}'.format(np.sum(chip_fake_binary == 255)), 'sure_fg', ], ['rgb',  'gray', 'gray'], (15,15))
    contours, _ = cv.findContours(sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return chip_fake_binary, sure_fg, dist_transform, contours


def getChipRes(
    img_chip: np.array,
    img_id: int,
    viz_res=False,
    debug=False,
) -> Dict:

    # results init
    chip_results = {"CR": 0, "CG": 0, "CB": 0, "CK": 0, "CW": 0}

    gray = cv.cvtColor(img_chip, cv.COLOR_RGB2GRAY)  # rgb to gray
    gray = cv.GaussianBlur(gray, (21, 21), 1)
    ret, binary = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )  # Otsu's thresholding after Gaussian filtering

    # In case of no chips
    if np.sum(binary == 255) > 1e6:
        return chip_results

    # first process RGB chips
    for chip_type in ["CR", "CG", "CB"]:
        image = cv.cvtColor(img_chip, cv.COLOR_RGB2HSV)
        lower1, upper1 = HSV_range(chip_type)
        lower_mask = cv.inRange(image, lower1, upper1)

        if chip_type == "CR":
            lower2 = lower1 + np.array([160, 0, 0])
            upper2 = upper1 + np.array([170, 0, 0])
            upper_mask = cv.inRange(image, lower2, upper2)
            full_mask = lower_mask + upper_mask

        else:
            full_mask = lower_mask

        # give fake images
        # chip_fake_im = copy.deepcopy(table[1000:2800, 2200:3900])
        # chip_fake_im[np.where(full_mask == 255)] = img_chip[np.where(full_mask == 255)]

        # give better masks
        kernelClose = np.ones((15, 15), np.uint8)
        full_mask = cv.morphologyEx(
            full_mask, cv.MORPH_CLOSE, kernelClose, iterations=2
        )

        # if with coins
        if np.sum(full_mask == 255) > 38000:
            chip_fake_binary, sure_fg, dist_transform, contours = RGB_chip_identify(
                img_chip, use_full_mask=True, full_mask=full_mask
            )

            if debug:
                plotMultipleImages(
                    1,
                    5,
                    images=[img_chip, full_mask, binary, dist_transform, sure_fg],
                    titles=[
                        'ori imag',
                        'full_mask {}'.format(np.sum(full_mask == 255)),
                        'binary',
                        'dist_transform',
                        'sure_fg{}'.format(img_id),
                    ],
                    cmap=['rgb', 'gray', 'gray', 'gray', 'gray'],
                    figsize=(20, 10),
                )

        else:
            # if no chips
            contours = []

            if debug:
                plotMultipleImages(
                    1,
                    2,
                    images=[img_chip, full_mask],
                    titles=[
                        'ori imag',
                        'full_mask {}'.format(np.sum(full_mask == 255)),
                    ],
                    cmap=['rgb', 'gray'],
                    figsize=(10, 10),
                )

        # count_areas = []
        for contour in contours:
            if cv.contourArea(contour) > 3800:
                # count_areas.append(cv.contourArea(contour))
                chip_results[chip_type] += 1

    for chip_type in ["CK"]:
        chip_fake_binary, sure_fg, dist_transform, contours = RGB_chip_identify(
            img_chip, use_full_mask=False
        )

        if debug:
            plotMultipleImages(
                1,
                4,
                images=[img_chip, chip_fake_binary, sure_fg, dist_transform],
                titles=[
                    'ori imag',
                    'chip_fake_binary {}'.format(img_id),
                    'sure_fg {}'.format(img_id),
                    'dist_transform {}'.format(img_id),
                ],
                cmap=['rgb', 'gray', 'gray', 'gray'],
                figsize=(20, 10),
            )

        total_chip = 0
        for contour in contours:
            if cv.contourArea(contour) > 3800:
                # count_areas.append(cv.contourArea(contour))
                total_chip += 1

        # print(chip_type, total_chip)
        chip_results[chip_type] = (
            total_chip - chip_results["CR"] - chip_results["CG"] - chip_results["CB"]
        )

    for chip_type in ["CW"]:
        image = cv.cvtColor(img_chip, cv.COLOR_RGB2HLS)
        lower1, upper1 = HSV_range(chip_type)
        lower_mask = cv.inRange(image, lower1, upper1)
        full_mask = lower_mask

        # if with coins
        if np.sum(full_mask == 255) > 38000:
            kernelClose = np.ones((3, 4), np.uint8)
            opening = cv.morphologyEx(full_mask, cv.MORPH_OPEN, (5, 5), iterations=2)
            closing = cv.morphologyEx(
                opening, cv.MORPH_CLOSE, kernelClose, iterations=1
            )

            dist_transform = cv.distanceTransform(closing, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(
                dist_transform, 0.4 * dist_transform.max(), 255, cv.THRESH_BINARY
            )
            sure_fg = np.uint8(sure_fg)

            contours, _ = cv.findContours(
                sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            if debug:
                plotMultipleImages(
                    1,
                    5,
                    images=[img_chip, full_mask, binary, dist_transform, sure_fg],
                    titles=[
                        'ori imag',
                        'full_mask {}'.format(np.sum(full_mask == 255)),
                        'binary',
                        'dist_transform',
                        'sure_fg{}'.format(img_id),
                    ],
                    cmap=['rgb', 'gray', 'gray', 'gray', 'gray'],
                    figsize=(20, 20),
                )

        else:
            # if no chips
            contours = []

            if debug:
                plotMultipleImages(
                    1,
                    2,
                    images=[img_chip, full_mask],
                    titles=[
                        'ori imag',
                        'full_mask {}'.format(np.sum(full_mask == 255)),
                    ],
                    cmap=['rgb', 'gray'],
                    figsize=(10, 10),
                )

        # count_areas = []
        for contour in contours:
            if cv.contourArea(contour) > 3800:
                # count_areas.append(cv.contourArea(contour))
                chip_results[chip_type] += 1

    if viz_res:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_chip)
        ax.set_title('Img' + str(img_id) + str(chip_results).replace(" ", ""))

    return chip_results
