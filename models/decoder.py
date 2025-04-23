import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torchvision import transforms as T


class Inversion(nn.Module):
    def __init__(self):
        super(Inversion, self).__init__()

        self.device = torch.device('cuda')
        self.std = 0.001 # imagenet 0.01, cifar 0.001

        # kernel size: imagenet 3, cifar 9
        self.convtrans1 = nn.ConvTranspose2d(64, 64, 9, stride=1, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 3,  9, stride=1, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)

    def get_mask(self, shape, pos):
        masks = []
        for c in range(shape[1]):
            mask = []
            for i in range(shape[3]):
                row = []
                for j in range(shape[2]):
                    sub = np.zeros(4)
                    sub[pos] = 1
                    row.append(sub.reshape((2, 2)))
                row = np.concatenate(row, axis=1)
                mask.append(row)
            mask = np.concatenate(mask, axis=0)
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def forward(self, x):
        x = self.upsample(x)
        h, w = x.shape[2:]

        saw = Variable(torch.empty(x.shape, device=x.device)\
                        .normal_(mean=0.0, std=self.std))
        mask = torch.from_numpy(self.get_mask((1, 64, h//2, w//2), 0))\
                        .float().to(x.device)

        x = x * mask + saw * (1 - mask)
        x = F.relu(self.convtrans1(x))
        x = self.convtrans2(x)
        x = torch.clamp(unnormalize(x), min=0.0, max=1.0)

        return x


def unnormalize(x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std  = torch.FloatTensor([0.229, 0.224, 0.225])
    unnorm = T.Normalize(- mean / std, 1 / std)
    return unnorm(x)
