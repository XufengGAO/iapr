import os
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

curr_dir_path = os.path.dirname(os.path.realpath("__file__"))
model_num_path = os.path.join(
    curr_dir_path, 'mlp_classification', 'model_num_2022-06-03-06-33.pt'
)
model_sb_path = os.path.join(
    curr_dir_path, 'mlp_classification', 'model_sb_2022-06-03-06-51.pt'
)

# MLP Model
class MLP(torch.nn.Module):
    def __init__(self, classes=14):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(3000, 500)
        self.fc3 = torch.nn.Linear(500, 100)
        self.fc4 = torch.nn.Linear(100, classes)

    def forward(self, input):
        input = torch.nn.functional.relu(self.fc1(input))
        input = self.fc3(input)
        input = self.fc4(input)
        return input


def num_sb_classification(
    ims,
    model_num_path=model_num_path,
    model_sb_path=model_sb_path,
    debug=False,
):

    i = 0
    im_num = Image.fromarray(ims[i].astype(np.uint8)).resize([50, 60])

    im_num = im_num.convert("L")
    im_num = torch.from_numpy(np.array(im_num).flatten().astype(np.float32))

    i = 1
    im_sb = Image.fromarray(ims[i].astype(np.uint8)).resize([50, 60])

    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(im_num)
        ax[1].imshow(im_sb)

    im_sb = im_sb.convert("L")
    im_sb = torch.from_numpy(np.array(im_sb).flatten().astype(np.float32))

    model_num = MLP(classes=14)
    # path_num = '/content/drive/MyDrive/iapr/Models/num_2022-06-03-06-33.pt'
    model_num.load_state_dict(torch.load(model_num_path))
    # mn = model_num.eval()
    # print('mn:',mn)

    model_sb = MLP(classes=5)
    # path_sb = '/content/drive/MyDrive/iapr/Models/sb_2022-06-03-05-51.pt'
    model_sb.load_state_dict(torch.load(model_sb_path))
    # ms = model_sb.eval()
    # print('ms:',ms)

    outputs_num = model_num(im_num)
    _, pred_num = torch.max(outputs_num.data, dim=0)
    # print()
    outputs_sb = model_sb(im_sb)
    _, pred_sb = torch.max(outputs_sb.data, dim=0)
    # print(pred_sb.item())

    num_res = inv_num_dic[str(pred_num.item())]
    sb_res = inv_sb_dic[str(pred_sb.item())]

    if num_res == 'N' and sb_res == 'N':
        return '0'

    return num_res + sb_res


inv_num_dic = {
    '0': 'A',
    '1': '2',
    '2': '3',
    '3': '4',
    '4': '5',
    '5': '6',
    '6': '7',
    '7': '8',
    '8': '9',
    '9': '10',
    '10': 'J',
    '11': 'Q',
    '12': 'K',
    '13': 'N',
}


# sb_full_dict_iapr={'D':0,'H':1,'S':2,'C':3,'u':4}
inv_sb_dic = {'0': 'D', '1': 'H', '2': 'S', '3': 'C', '4': 'N'}


# dictonary from labels(str) to int
# num_full_dict_iapr={
# 'AS': 0, '2S':1, '3S':2, '4S':3, '5S':4, '6S':5, '7S':6,
# '8S':7, '9S':8, '10S':9, 'JS':10, 'QS':11, 'KS':12,
# 'AH': 0, '2H': 1, '3H': 2, '4H': 3, '5H': 4, '6H': 5,
# '7H':6, '8H':7, '9H':8, '10H':9, 'JH': 10, 'QH': 11,
# 'KH':12,
# 'AC': 0, '2C': 1, '3C': 2, '4C': 3, '5C': 4, '6C': 5,
# '7C':6, '8C':7, '9C':8, '10C':9, 'JC': 10, 'QC': 11,
# 'KC':12,
# 'AD': 0, '2D': 1, '3D': 2, '4D': 3, '5D': 4, '6D': 5,
# '7D':6, '8D':7, '9D':8, '10D':9, 'JD': 10, 'QD': 11,
# 'KD':12, 'null':13}
