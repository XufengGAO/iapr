#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# =============================================================================
"""
@Author        :   Yujie He
@File          :   test_process_image.py
@Date created  :   2022/06/03
@Maintainer    :   Yujie He
@Email         :   yujie.he@epfl.ch
"""
# =============================================================================
"""
The module provides test script for process all images from [train/test] data folder
Usage:
python test_process_image.py --mode train
python test_process_image.py --mode test
"""
# =============================================================================

import os
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

import argparse
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np

from preprocess_utils import cropImgParts, PART_NAMES, cropTable
from chip_utils import getChipRes
from card_utils import checkNoPlay, extractTableCard, getPlayerRes, checkPlaying
from utils import getGameDict
from yolodetector.detect import detectFromNp, loadYoloModel
from detect_utils import detectTableCard, detectPlayerCard
from utils import (
    eval_listof_games,
    debug_listof_games,
    save_results,
    load_results,
    eval_listof_games_custom,
)
from viz_utils import vizGameScores


def process_image(
    file: str,
    debug: bool = False,
    viz_parts: bool = False,
    verbose: bool = False,
    rect_offset: int = 70,
) -> Dict:
    """Process image and return information. To return the value of the cards we use
    the following format: {number}{color}. Where
        - color is either (D)imanond, (H)eart, (S)pade, (C)lub
        - number is either 2-10, (J)ack, (Q)ueen, (K)ing. A(ce).

    Args:
        file (str): Input image to process
        debug (bool, optional): debug flag. Defaults to False.
        viz_parts (bool, optional): viz cropped parts flag. Defaults to False.
        verbose (bool, optional): verbose flag. Defaults to False.
        rect_offset (int, optional): rectangle offset when cropping table. Defaults to 70.

    Returns:
        Dict: results of chips, table cards, and player cards
    """
    img_id = file.split('/')[-1][:-4]
    # step1: crop table from origin image
    table_crop, rect, rect_origin, imgs_debug = cropTable(
        file,
        kernel_sz=3,
        dist_thresh=0.2,
        edge_dilate_nr=5,
        use_convex_hull=False,
        resize_flag=True,
        resize_scale=4.0,
        crop_sz=(800, 0),
        rect_offset=rect_offset,
        debug=debug,
    )
    # step: crop different image parts
    img_parts = cropImgParts(
        table_crop,
        viz_parts=viz_parts,
        viz_inplace=debug,
        fig_title='Image ' + str(img_id),
    )
    # step3: detect chips number and color
    img_chip_np = np.asarray(img_parts[5])
    chip_res = getChipRes(
        img_chip=img_chip_np,
        img_id=img_id,
        viz_res=debug,
        debug=False,
    )
    if verbose:
        print(chip_res)

    # step4: detect T1-T5 cards with YOLO detector
    img_part_t = np.asarray(img_parts[4])
    img_table_cards = extractTableCard(
        img_part_t=img_part_t, use_alpha=False, debug=debug
    )
    model = loadYoloModel()
    t15_res = detectTableCard(img_table_cards, model, debug=debug)
    if verbose:
        print(t15_res)

    # step5: check not-playing ids -> List[bool]
    img_players_np = [np.asarray(im) for im in img_parts[:4]]
    # playing_status = checkNoPlay(
    #     img_players=img_players_np,
    #     intensity_thresh=0.8,
    #     red_thresh=0.2,
    #     blue_thresh=0.023,
    #     debug=debug,
    #     fig_title=img_id,
    # )
    playing_status = checkPlaying(img_players_np, debug=debug)
    if verbose:
        print(playing_status)

    # step6: detect players' cards
    # player_res = detectPlayerCards(playing_status)
    img_player_cards = getPlayerRes(
        img_players=img_players_np,
        img_ID=img_id,
        debug=debug,
    )
    player_res = detectPlayerCard(
        img_player_cards=img_player_cards,
        playing_status=playing_status,
        model=model,
        debug=debug,
    )
    if verbose:
        print(player_res)

    # step7: merge all results
    results = {**t15_res, **player_res, **chip_res}

    return results


def naive_main():
    res = process_image(
        'data/train/train_22.jpg',
        debug=False,
        verbose=False,
    )
    print(res)

def main(mode: str = "train"):
    # test | train
    project_dir_path = os.path.dirname(os.path.realpath(__file__))
    path_data = os.path.join(project_dir_path, "data", mode)
    group_id = 32

    # data_sz = 28
    data_sz = np.loadtxt(os.path.join(path_data, mode+"_size.txt")).astype(int)

    # Load images from folder
    game_results = {}

    # Evaluate three images
    games_id = [i for i in range(data_sz)]  # to evaluate images

    # for i in games_id:
    for i in tqdm(range(data_sz)):
        file = os.path.join(
            path_data, "{}_{}.jpg".format(mode, str(i).zfill(2))
        )  # Fill the string with zeros until it is 2 characters long
        # open the image
        results = process_image(file=file)
        # Append result to array
        game_results[i] = results

        file_results = save_results(results=game_results, groupid=group_id)

    loaded_results = load_results(file_results)
    print("Evaluated games", list(loaded_results.keys()))

    if mode == 'train':
        # Read training data
        game_labels = pd.read_csv('data/train/updated_train_labels.csv')
        game_labels = game_labels.fillna('0')

        avg_score, score_list = eval_listof_games_custom(game_results , game_labels , game_id = games_id)
        vizGameScores(avg_score, score_list, data_sz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default='test',
        choices=['test', 'train'],
        type=str,
        help="data mode"
        )
    args = parser.parse_args()
    main(args.mode)
