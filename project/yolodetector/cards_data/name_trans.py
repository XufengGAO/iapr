from torch import import_ir_module


namesfile = r'C:\Users\Jugg\Desktop\PlaycDC\YOLO\cards_data\cards.names'
namesfile_trans = r'C:\Users\Jugg\Desktop\PlaycDC\YOLO\cards_data\cards_iapr.names'

class_names = []
with open(namesfile, 'r', encoding='utf8') as fp:
    lines = fp.readlines()
for line in lines:
    class_names.append(line.strip().upper())
    with open(namesfile_trans,'a') as f:    #设置文件对象
        f.write(line.strip().upper())
        f.write('\n')      


           #将字符串写入文件中
import pdb
pdb.set_trace()