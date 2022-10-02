import warnings
import os
import re
import yaml
import argparse
from net.net import *
from data_pipeline import FORCE_Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='classification_data/input_feature',
                        help='path for input features files')
    parser.add_argument('-o', '--output_dir', type=str, default='classification_data/output_feature',
                        help='path for output features files')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoints/LeNET_checkpoint_final.pth',
                        help='checkpoint path')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    input_root = args.input_dir
    output_root = args.output_dir

    output_dir = './test_result'
    os.makedirs(output_dir, exist_ok=True)

    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {"LeNET": LeNet}
    model_name = cfg['MODEL']['TYPE']  # 模型类型

    model = model_dict[model_name]().to(device).eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    test_list = np.load("test_list.npy")
    test_list = [str(i) for i in test_list]
    random.shuffle(test_list)
    test_set = FORCE_Dataset(input_root, output_root, test_list)

    testDataLoader = DataLoader(test_set, batch_size=1)

    with torch.no_grad():
        acc_num = 0
        total_num = 0
        for index, sample in enumerate(testDataLoader):
            total_num += 1
            dats = sample['X'].to(device).float()
            label = sample['y'].to(device).double()

            output = model(dats).cpu()
            label = label.cpu()
            if label[0] == np.argmax(output):
                acc_num += 1

            interval = 1

            if (index + 1) % interval == 0:
                print('Processed {} / {} : {}|{}'.format(index + 1, len(test_list), int(label[0]),
                                                          np.argmax(output)))

        print("{}%".format(acc_num / total_num * 100))


if __name__ == '__main__':
    main()
