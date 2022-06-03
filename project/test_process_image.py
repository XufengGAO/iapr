import warnings

warnings.filterwarnings('ignore')

from typing import Dict

from PIL import Image
import numpy as np

from preprocess_utils import cropImgParts, PART_NAMES, cropTable
from chip_utils import getChipRes
from card_utils import checkNoPlay, extractTableCard, getPlayerRes, checkPlaying
from utils import getGameDict
from yolodetector.detect import detectFromNp, loadYoloModel
from detect_utils import detectTableCard, detectPlayerCard


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
        crop_sz=(800, 20),
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


def main():
    res = process_image(
        'data/train/train_22.jpg',
        debug=False,
        verbose=False,
    )
    print(res)


if __name__ == "__main__":
    main()
