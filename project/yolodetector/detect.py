# std
import sys, time, pdb
from pathlib import Path
from typing import Dict, Union

# imported
from PIL import Image
import torch.nn as nn

# custom
from yolodetector.detect_utils import *
from yolodetector.darknet import Darknet

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(curr_dir_path, 'yolov3-tiny.cfg')
WEIGHT = os.path.join(curr_dir_path, 'backup/hardest.weights')
NAMES = os.path.join(curr_dir_path, 'cards_data/cards_iapr.names')


def loadYoloModel(
    cfgfile: Union[str, Path] = CFG,
    weightfile: Union[str, Path] = WEIGHT,
    use_cuda: bool = True,
) -> nn.Module:
    """_summary_

    Args:
        cfgfile (Union[str, Path], optional): _description_. Defaults to CFG.
        weightfile (Union[str, Path], optional): _description_. Defaults to WEIGHT.
        use_cuda (bool, optional): _description_. Defaults to True.

    Returns:
        nn.Module: _description_
    """
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        model.cuda()
    return model


def detectFromFile(
    imgfile: Union[str, Path],
    model: nn.Module,
    namesfile: Union[str, Path] = NAMES,
) -> Dict:
    """Method that performs prediction given image path and a weightfile

    Args:
        imgfile (Union[str, Path]): _description_
        model (nn.Module): _description_
        namesfile (Union[str, Path], optional): _description_. Defaults to NAMES.

    Returns:
        Dict: detected results with {res: confidence} like dictionary
    """
    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))
    # start = time.time()
    boxes = do_detect(model, sized, 0.1, 0.1)
    # finish = time.time()
    # print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    class_names = load_class_names(namesfile)
    # cls_conf = box[5]
    # cls_id = box[6]
    detected_res = {
        class_names[box[6]]: box[5].item() for box in boxes if len(box) >= 7
    }
    # _, cls_dict = plot_boxes(img, boxes, None, class_names)

    return detected_res


def detectFromNp(
    img: Image,
    model: nn.Module,
    namesfile: Union[str, Path] = NAMES,
) -> Dict:
    """Method that performs prediction given an PIL image and a weightfile

    Args:
        imgfile (Union[str, Path]): _description_
        model (nn.Module): _description_
        namesfile (Union[str, Path], optional): _description_. Defaults to NAMES.

    Returns:
        Dict: detected results with {res: confidence} like dictionary
    """
    sized = img.resize((model.width, model.height))
    boxes = do_detect(model, sized, 0.1, 0.1)

    class_names = load_class_names(namesfile)
    # cls_conf = box[5]
    # cls_id = box[6]
    detected_res = {class_names[box[6]]: box[5].item() for box in boxes}

    return detected_res


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Example: python -m yolodetector.detect test_2.png
        imgfile = sys.argv[1]  # source img file
        model = loadYoloModel(cfgfile=CFG, weightfile=WEIGHT, use_cuda=True)
        detected_res = detectFromFile(
            imgfile,
            model,
            NAMES,
        )
        print("Estimated:", detected_res)
    elif len(sys.argv) == 4:
        # Example: python yolodetector/detect.py yolodetector/yolov3-tiny.cfg yolodetector/backup/hardest.weights test_2.png
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        globals()["namesfile"] = NAMES
        model = loadYoloModel(cfgfile=CFG, weightfile=WEIGHT, use_cuda=True)
        detected_res = detectFromFile(
            imgfile,
            model,
            NAMES,
        )
        print("Estimated:", detected_res)
    else:
        print('Usage: ')
        print('  python detect.py imgfile')
