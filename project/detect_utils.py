from typing import List

import torch.nn as nn
import numpy as np
import imutils
import PIL

from yolodetector.detect import detectFromNp
from viz_utils import plotMultipleImages
from preprocess_utils import addContrast


def detectTableCard(
    img_table_cards: List[np.array],
    model: nn.Module,
    debug: bool = False,
):
    t15_list = []
    for idx, img_table_card in enumerate(img_table_cards):

        # img_table_card = addContrast(img_table_card)
        img_table_card_rot = imutils.rotate_bound(img_table_card, 180)

        test_img = PIL.Image.fromarray(img_table_card)
        test_img_rot = PIL.Image.fromarray(img_table_card_rot)

        # conf: id
        detected_res_ = detectFromNp(
            test_img,
            model,
        )
        detected_res_rot_ = detectFromNp(
            test_img_rot,
            model,
        )
        detected_res = {**detected_res_, **detected_res_rot_}
        # print(idx, detected_res)
        if detected_res != {}:
            # str_code = max(detected_res, key=detected_res.get)
            max_conf = max(detected_res.keys())
            str_code = detected_res[max_conf]
        else:
            str_code = '0'
        t15_list.append(str_code)
    t15_res = {'T' + str(idx + 1): t15_list[idx] for idx in range(len(t15_list))}
    if debug:
        print(t15_res)
        plotMultipleImages(
            1,
            5,
            images=img_table_cards,
            titles=t15_list,
            cmap=['rgb'] * 5,
            figsize=(20, 6),
        )
    return t15_res


def detectPlayerCard(
    img_player_cards: List[np.array],
    playing_status: List[bool],
    model: nn.Module,
    debug: bool = False,
):
    player_list = []
    for idx, img_player_card in enumerate(img_player_cards):

        if playing_status[idx // 2]:
            test_img = PIL.Image.fromarray(img_player_card)
            detected_res = detectFromNp(
                test_img,
                model,
            )
            if detected_res != {}:
                # str_code = max(detected_res, key=detected_res.get)
                max_conf = max(detected_res.keys())
                str_code = detected_res[max_conf]
            else:
                str_code = '0'
        else:
            str_code = '0'
        player_list.append(str_code)
    player_res = {
        'P' + str(idx // 2 + 1) + str(idx % 2 + 1): player_list[idx]
        for idx in range(len(player_list))
    }
    if debug:
        print(player_res)
        plotMultipleImages(
            2,
            4,
            images=img_player_cards,
            titles=player_list,
            cmap=['rgb'] * 8,
            figsize=(20, 10),
        )
    return player_res
