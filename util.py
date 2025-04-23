import numpy as np
import torch
import torch.nn.functional as F

from models.densenet import densenet121, densenet161, densenet169
from models.googlenet import googlenet
from models.inception import inception_v3
from models.resnet import resnet18, resnet34, resnet50
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torch.utils.data import DataLoader, Subset
from torchvision import models as M
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, ImageNet

import os


_cifar_networks = {
    'vgg11_bn_1':   vgg11_bn(),
    'vgg13_bn_1':   vgg13_bn(),
    'vgg16_bn_1':   vgg16_bn(),
    'vgg19_bn_1':   vgg19_bn(),
    'resnet18':     resnet18(),
    'resnet34':     resnet34(),
    'resnet50':     resnet50(),
    'densenet121':  densenet121(),
    'densenet161':  densenet161(),
    'densenet169':  densenet169(),
    'googlenet':    googlenet(),
    'inception_v3': inception_v3(),
}

_imagenet_networks = {
    'resnet18':             M.resnet18(pretrained=True),
    'alexnet':              M.alexnet(pretrained=True),
    'squeezenet':           M.squeezenet1_0(pretrained=True),
    'vgg16':                M.vgg16(pretrained=True),
    'densenet':             M.densenet161(pretrained=True),
    'inception':            M.inception_v3(pretrained=True),
    'googlenet':            M.googlenet(pretrained=True),
    'shufflenet':           M.shufflenet_v2_x1_0(pretrained=True),
    'mobilenet_v2':         M.mobilenet_v2(pretrained=True),
    'mobilenet_v3_large':   M.mobilenet_v3_large(pretrained=True),
    'mobilenet_v3_small':   M.mobilenet_v3_small(pretrained=True),
    'resnext50_32x4d':      M.resnext50_32x4d(pretrained=True),
    'wide_resnet50_2':      M.wide_resnet50_2(pretrained=True),
    'mnasnet':              M.mnasnet1_0(pretrained=True),
    'efficientnet_b0':      M.efficientnet_b0(pretrained=True),
    'efficientnet_b7':      M.efficientnet_b7(pretrained=True),
    'regnet_y_16gf':        M.regnet_y_16gf(pretrained=True),
    'regnet_y_32gf':        M.regnet_y_32gf(pretrained=True),
    'regnet_x_800mf':       M.regnet_x_800mf(pretrained=True),
    'regnet_x_1_6gf':       M.regnet_x_1_6gf(pretrained=True),
}

_mean = {
    'default':  [0.5   , 0.5   , 0.5   ],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'imagenet': [0.485 , 0.456 , 0.406 ],
}

_std = {
    'default':  [0.5   , 0.5   , 0.5   ],
    # 'cifar10':  [0.2471, 0.2435, 0.2616],   # huyvnphan/PyTorch_CIFAR10
    'cifar10':  [0.2023, 0.1994, 0.201],    # chenyaofo/pytorch-cifar-models
    'imagenet': [0.229 , 0.224 , 0.225],
}

_size = {
    'cifar10':  ( 32,  32, 3),
    'imagenet': (224, 224, 3),
}

_num = {
    'cifar10':  10,
    'imagenet': 1000,
}


def get_norm(dataset):
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = T.Normalize(mean, std)
    unnormalize = T.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_data(loader, source, size=100):
    x_data = []
    y_data = []
    for i in range(15):
        x_batch, y_batch = loader.get_next_batch()
        indices = np.where(y_batch == source)[0]
        if i == 0:
            x_data = x_batch[indices]
            y_data = y_batch[indices]
        else:
            x_data = np.concatenate((x_data, x_batch[indices]), axis=0)
            y_data = np.concatenate((y_data, y_batch[indices]), axis=0)
        if x_data.shape[0] >= size:
            break

    x_data = x_data[:size] / 255.
    y_data = y_data[:size]
    print('data:', x_data.shape, y_data.shape)

    return x_data, y_data


def get_loader(dataset, train, batch_size, ratio=1.0):
    if dataset == 'cifar10':
        transform = T.Compose([T.ToTensor(),
                               T.Normalize(_mean[dataset], _std[dataset])])
        dataset = CIFAR10(root='./data', train=train, transform=transform,download=True)
    elif dataset == 'imagenet':
        split = 'train' if train else 'val'
        transform = T.Compose([T.Resize(256),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               T.Normalize(_mean[dataset], _std[dataset])])
        dataset = ImageNet('data/imagenet', split=split, transform=transform)

    if ratio < 1:
        indices = np.arange(int(len(dataset) * ratio))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            shuffle=train,
                            pin_memory=True,
                            drop_last=train) # for retrain
    return dataloader


def get_model(dataset, network):
    if dataset == 'cifar10':
        # huyvnphan/PyTorch_CIFAR10
        # model = _cifar_networks[network]
        # model.load_state_dict(torch.load(f'ckpt/{network}.pt'))

        # chenyaofo/pytorch-cifar-models
        model_name = f'cifar10_{network}'
        model = torch.hub.load('chenyaofo/pytorch-cifar-models',
                               model_name,
                               pretrained=True)
    elif dataset == 'imagenet':
        model = _imagenet_networks[network]
        if 'eps' in network:
            prefix = 'module.model.' if 'resnet' in network\
                        else 'module.model.model.'
            state_dict = torch.load(f'ckpt/{network}.ckpt')['model']
            state_dict = {k[len(prefix):]:v for k, v in state_dict.items()\
                            if 'attacker' not in k and 'normalizer' not in k}
            model.load_state_dict(state_dict)
    return model


def get_classes(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def replacezero(t):
    z = torch.ones((), device=t.device, dtype=t.dtype)
    t = torch.where(t == 0, z, t)
    return t


def TV_loss(x):
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def composite_trigger(inputs, pattern):
    t = inputs.size(3) // 2
    pattern = pattern.repeat(inputs.size(0), 1, 1, 1)
    out = torch.cat([pattern[:, :, :, :t], inputs[:, :, :, t:]], dim=3)
    return out


def wanet_trigger(inputs, grids, channel_bias, threshold=0.01):
    # inputs.shape = [batch_size, 3, 32, 32]
    # grid.shape = [1, 27, kdim, kdim]
    # channel_bias.shape = [1, 3, 1, 1]
    height = inputs.size(2)
    grids = F.upsample(grids, size=height, mode='bicubic', align_corners=True)\
            .permute(0, 2, 3, 1).view(height, height, 3, 3, 3) # (32, 32, 3, 3, 3)

    if channel_bias.size(2) != 1:
        channel_bias = F.upsample(channel_bias, size=height, mode='bicubic',
                                  align_corners=True)

    k = 3 # kernel_size
    n, c, h, w = inputs.size()
    device = inputs.device

    # Center set 1
    grid_bias = torch.zeros((3, 3))
    grid_bias[1, 1] = 1
    grid_bias = grid_bias.repeat(h, w, 3, 1, 1).to(device) # [32, 32, 3, 3, 3]

    # Constrain the threshold
    grids = torch.clamp(grids, -threshold, threshold)
    grids += grid_bias
    channel_bias = torch.clamp(channel_bias, -threshold, threshold).to(device)

    pad_inputs = F.pad(inputs, (1, 1, 1, 1), "constant", 0)
    data = F.unfold(pad_inputs, (k, k))
    data = data.permute(0, 2, 1)
    data = data.view(n, h, w, c, k, k) # [batch_size, 32, 32, 3, 3, 3]
    grids = grids[None, :, :, :, :, :] # [1, 32, 32, 3, 3, 3]
    out = (data * grids).sum(dim=[-2, -1]).permute(0, 3, 1, 2)
    out = torch.clamp(out + channel_bias, 0., 1.)
    return out
