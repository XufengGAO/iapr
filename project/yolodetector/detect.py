from pydoc import classname
import sys, time, pdb

from PIL import Image, ImageDraw
sys.path.append(r'C:\Users\Jugg\Desktop\iapr\project\PlaycDC\YOLO')
from yolodetector.detect_utils import *
from yolodetector.darknet import Darknet

namesfile=None

def card_detect(imgfile, cfgfile='yolov3-tiny.cfg', weightfile='backup/hardest.weights', namesfile='cards_data/cards_iapr.names'):
    # pdb.set_trace()
    startTime=time.strftime("%y-%m-%d-%H-%M", time.localtime()) 
    """Method that performs prediction given an image and a weightfile"""
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    use_cuda = False
    if use_cuda:
        model.cuda()
    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((model.width, model.height))
    start = time.time()
    boxes = do_detect(model, sized, 0.1, 0.1, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))
    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions'+startTime+'.jpg', class_names)
    return class_names

if __name__ == '__main__':
    # pdb.set_trace()
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1] # configuration settings 'cards_data/yolov3-tiny.cfg'
        weightfile = sys.argv[2] # 'backup/hardest.weights'
        imgfile = sys.argv[3] # source img file
        # globals()["namesfile"] = sys.argv[4] # names dictonary
        card_detect('cards_data/yolov3-tiny.cfg', 'backup/hardest.weights', imgfile, namesfile='cards_data/cards_iapr.names')
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile names')
