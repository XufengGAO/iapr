from typing import List
import numpy as np
import cv2 as cv
from viz_utils import plotMultipleImages


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
