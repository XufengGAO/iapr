import PIL
from yolodetector.detect import detectFromNp
from viz_utils import plotMultipleImages


def detectTableCard(img_table_cards, model, debug=False):
    t15_list = []
    for idx, img_table_card in enumerate(img_table_cards):

        test_img = PIL.Image.fromarray(img_table_card)
        detected_res = detectFromNp(
            test_img,
            model,
        )
        str_code = max(detected_res, key=detected_res.get)
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
