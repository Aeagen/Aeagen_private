import argparse

from data import MyDataset
from netwrok import ResNet50Features, BiomedCLIP, Cov_Att
import torch
from torch.utils.data import DataLoader
import torch.nn.init as init
from labels_num import label_map
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle
import pickletools
import re
from typing import List, Optional, Tuple, Union, io
import io
from torch import nn

import legacy
import click
import dnnlib
import numpy as np
import PIL.Image
import torch

from torch_utils import gen_utils


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


# ----------------------------------------------------------------------------
def load_g_ema(network_pkl):
    with open(network_pkl, 'rb') as handle:
        d_starts = -1
        g_ema_starts = -1
        for i, op in enumerate(pickletools.genops(handle)):
            if op[0].name == "SHORT_BINUNICODE":
                if op[1] == 'G_ema':
                    g_ema_starts = op[2]
                elif op[1] == 'D':
                    d_starts = op[2]
        assert d_starts >= 0 and g_ema_starts >= 0

    with open(network_pkl, 'rb') as handle:
        bs = handle.read()
        bs = bs[:d_starts] + bs[g_ema_starts:]
        obj = pickle.Unpickler(io.BytesIO(bs)).load()

    return obj['G_ema']


def load_styleGAN(
        device
):
    network_pkl = 'prenetwork_weight/radimagegan64x64.pkl'
    print('Loading StyleGan-xl from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = load_g_ema(network_pkl)
        G = G.eval().requires_grad_(False).to(device)
    return G


def generate_images(
        seed,
        device,
        G,
        batch_size,
        cls
):
    class_idx = seed
    batch_sz = batch_size
    truncation_psi = 0.7
    centroids_path = None
    noise_mode = 'const'
    translate = parse_vec2('0,0')
    rotate = 0

    # Generate images.
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    w = gen_utils.get_w_from_seed(G, batch_sz, device, clss=cls, truncation_psi=truncation_psi, seed=class_idx,
                                  centroids_path=centroids_path, class_idx=class_idx)
    img = gen_utils.w_to_img(G, w, to_np=True)
    return img


def init_conv(m):
    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
        init.constant_(m.bias, 0)


def init_norm(m):
    init.constant_(m.weight, 1)
    init.constant_(m.bias, 0.01)


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./MIMIC-CXR', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--size', type=int, default=224, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--load_model_weight', type=bool, default=False, help='loading the weight of the model')
parser.add_argument('--seed', type=int, default=10, help='the random number of the numpy of input image')
parser.add_argument('--momentum', type=float, default=0.9, help='参数在优化算法中用于加速收敛过程，可以有效地减少震荡')

if __name__ == '__main__':
    opt = parser.parse_args()

    transform_args = dict(
        resize=opt.size,
        crop_size=opt.size,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    dataset = MyDataset(opt.dataroot, transform_args, label_map)

    # 构建DataLoader
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    # 判斷是否使用GPU進行運算
    if opt.cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print(device)
    # 创建一个 ResNet50 特征提取网络实例，并加载权重
    resnet50_features_model = ResNet50Features().to(device)
    # 在预训练模式下进行推断
    resnet50_features_model.eval()

    # 創建文本編碼器
    Text_CLIP = BiomedCLIP().to(device)

    for param in resnet50_features_model.parameters():
        param.requires_grad = False
    for param in Text_CLIP.parameters():
        param.requires_grad = False

    # 創建模型
    Model = Cov_Att().to(device)

    # 初始化Model的权重
    for layer in Model.modules():
        if isinstance(layer, nn.Conv2d):
            init_conv(layer)
        elif isinstance(layer, nn.BatchNorm2d):
            init_norm(layer)

    # 创建StyleGAN-xl模型
    G = load_styleGAN(device)

    # 定义优化器，只更新 Model 的参数
    optimizer = optim.SGD(Model.parameters(), lr=opt.lr, momentum=opt.momentum)
    losses = []
    best_loss = 100000
    # 训练循环
    for epoch in range(opt.n_epochs):
        print("第", epoch, "论训练")
        loss_ave = []
        for img, texts, labels in dataloader:
            labels = list(labels)
            numeric_labels = [label_map[label] for label in labels]

            texts = Text_CLIP.tokenizer(texts, Text_CLIP.context_length).to(device)
            img = img.to(device)

            # 清除之前计算的梯度
            optimizer.zero_grad()

            img = resnet50_features_model(img)
            text_encoder = Text_CLIP(texts)
            text_encoder = text_encoder.unsqueeze(1)

            img = Model(img, text_encoder)

            # 禁止梯度跟踪以确保生成图像时不会更新权重
            with torch.no_grad():
                gen_img = generate_images(seed=opt.seed,
                                          device=device,
                                          G=G,
                                          batch_size=opt.batchSize,
                                          cls=numeric_labels)

            out_img = img + gen_img

            sample_img = dataset.sample_random_images_by_labels(numeric_labels).to(device)
            # 计算 L2 损失
            loss = F.mse_loss(out_img, sample_img)

            # 反向传播并更新模型参数
            loss.backward()
            optimizer.step()

            loss_ave.append(loss.item())

        if (epoch + 1) % 10 == 0:
            torch.save(Model.state_dict(), f"model_epoch.pt")
            print("Model saved.")

        if (np.mean(loss_ave) < best_loss):
            torch.save(Model.state_dict(), f"best_model.pt")

        losses.append(np.mean(loss_ave))













