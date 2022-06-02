import sys, time, pdb
from pathlib import Path
from typing import Dict, Union
from PIL import Image

sys.path.append(r'C:\Users\Jugg\Desktop\iapr\project\PlaycDC\YOLO')
from yolodetector.detect_utils import *
from yolodetector.darknet import Darknet

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(curr_dir_path, 'yolov3-tiny.cfg')
WEIGHT = os.path.join(curr_dir_path, 'backup/hardest.weights')
NAMES = os.path.join(curr_dir_path, 'cards_data/cards_iapr.names')


def loadYoloModel(cfgfile=CFG, weightfile=WEIGHT, use_cuda=True):
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        model.cuda()
    return model


def detectFromFile(
    imgfile: Union[str, Path],
    cfgfile: Union[str, Path] = CFG,
    weightfile: Union[str, Path] = WEIGHT,
    namesfile: Union[str, Path] = NAMES,
    use_cuda: bool = True,
) -> Dict:
    """Method that performs prediction given an image and a weightfile

    Args:
        imgfile (Union[str, Path]): _description_
        cfgfile (Union[str, Path], optional): _description_. Defaults to CFG.
        weightfile (Union[str, Path], optional): _description_. Defaults to WEIGHT.
        namesfile (Union[str, Path], optional): _description_. Defaults to NAMES.
        use_cuda (bool, optional): _description_. Defaults to True.

    Returns:
        Dict: detected results with {res: confidence} like dictionary
    """
    model = loadYoloModel(cfgfile=cfgfile, weightfile=weightfile, use_cuda=True)
    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))
    # start = time.time()
    boxes = do_detect(model, sized, 0.1, 0.1, use_cuda)
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


def detectFromNp(img, cfgfile=CFG, weightfile=WEIGHT, namesfile=NAMES):
    """Method that performs prediction given an image and a weightfile"""
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    use_cuda = False
    if use_cuda:
        model.cuda()
    # img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))
    boxes = do_detect(model, sized, 0.1, 0.1, use_cuda)

    class_names = load_class_names(namesfile)
    # cls_conf = box[5]
    # cls_id = box[6]
    detected_res = {class_names[box[6]]: box[5].item() for box in boxes}

    return detected_res


if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Example: python -m yolodetector.detect test_2.png
        imgfile = sys.argv[1]  # source img file
        detected_res = detectFromFile(
            imgfile,
            CFG,
            WEIGHT,
            NAMES,
        )
        print("Estimated:", detected_res)
    elif len(sys.argv) == 4:
        # Example: python yolodetector/detect.py yolodetector/yolov3-tiny.cfg yolodetector/backup/hardest.weights test_2.png
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        globals()["namesfile"] = NAMES
        detected_res = detectFromFile(
            imgfile,
            cfgfile,
            weightfile,
        )
        print("Estimated:", detected_res)
    else:
        print('Usage: ')
        print('  python detect.py imgfile')
