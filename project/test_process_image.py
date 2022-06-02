import numpy as np
from preprocess_utils import cropImgParts, PART_NAMES, cropTable
from chip_utils import getChipRes
from card_utils import checkNoPlay, extractTableCard
# from data_utils import getGameDict
from yolodetector.detect import card_detect


def process_image(
    file,
    debug=False,
    viz_parts=False,
):
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
        rect_offset=50,
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
    # step4: detect T1-T5 cards
    img_part_t = np.asarray(img_parts[4])
    img_table_cards = extractTableCard(img_part_t=img_part_t, debug=debug)
    t15_results = {}

    card_detect(img_table_cards[0])

    # step5: check not-playing ids -> List[bool]
    img_players_np = np.asarray(img_parts[:4])
    playing_status = checkNoPlay(
        img_players=img_players_np,
        intensity_thresh=0.8,
        red_thresh=0.2,
        blue_thresh=0.023,
        debug=debug,
        fig_title=img_id,
    )

    # step6: detect players' cards
    # player_results = detectPlayerCards(playing_status)
    player_results = {}

    # step7: merge all results
    dummy_results = {**t15_results, **player_results, **chip_res}

    dummy_results = {
        # Flop, river and turn
        "T1": "2C",
        "T2": "AH",
        "T3": "8D",
        "T4": "JC",
        "T5": "JS",
        # Player cards
        "P11": '2H',
        "P12": '3C',
        "P21": 'KS',
        "P22": "QS",
        "P31": "KD",
        "P32": "5S",
        "P41": "7H",
        "P42": "8H",
        # Chips
        "CR": 1,
        "CG": 1,
        "CB": 1,
        "CK": 5,
        "CW": 0,
    }
    return dummy_results


def main():
    res = process_image('data/train/train_22.jpg', debug=True)
    print(res)


if __name__ == "__main__":
    main()
