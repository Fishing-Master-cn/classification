import warnings
import os
import re
import yaml
import argparse
import numpy as np
import torch
from skimage.io import imsave
from skimage.transform import resize
from net.net import *
from loader import FORCE_Dataset
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='grf_force_pre_data/test/input_feature',
                        help='path for input features files')
    parser.add_argument('-o', '--output_dir', type=str, default='grf_force_pre_data/test/output_feature',
                        help='path for output features files')
    parser.add_argument('-c', '--checkpoint', type=str, default='checkpoints/checkpoint_1.pth',
                        help='checkpoint path')
    return parser.parse_args()


def Compose(data):
    return data


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint
    input_root = args.image_dir
    output_root = args.mask_dir

    output_dir = './test_result'
    os.makedirs(output_dir, exist_ok=True)

    with open('./config.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {"Neural": NeuralNetwork, "Lstm": LstmRNN, "FCNs": FCNs}
    output_dim = cfg['TRAIN']['OUTPUT_DIM']
    model_name = cfg['MODEL']['TYPE']  # 模型类型

    model = model_dict[model_name](output_dim=output_dim).to(device).eval()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    test_list = [re.findall('(\d+)', file)[0] for file in os.listdir('./data/images')]
    test_set = FORCE_Dataset(input_root, output_root, test_list)

    with torch.no_grad():
        for index, (sample, data_id) in enumerate(test_set):
            test_data_batch = sample['X'].to(device).double()
            test_label_batch = sample['y'].to(device).double()

            output = torch.sigmoid(model(test_data_batch))
            # 将二维图转化为与 时间-力 图像保持一致
            pred_rescaled = Compose(output)
            label_rescaled = Compose(test_label_batch)

            plt.plot(pred_rescaled)
            plt.plot(label_rescaled)
            # plt.show()
            imsave(os.path.join(output_dir, '{}-image.png'.format(data_id)))

            interval = 1

            if (index + 1) % interval == 0:
                print('Processed {} / {}'.format(index + 1, len(test_list)))


if __name__ == '__main__':
    main()
