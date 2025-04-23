import numpy as np
import pytorch_msssim
import random
import sys
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models as M
from torchvision import transforms as T
from torchvision.io import read_image

from models.unet_model import UNet
from models.input_aware.models import Generator
from util import replacezero, TV_loss
from util import wanet_trigger, composite_trigger
from stylegan import StyleGAN, get_lr


class NC:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, asr_bound=0.99,
                 lr=0.1, init_cost=1e-3, normalize=None, augment=False):
        self.model = model
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.asr_bound = asr_bound
        self.lr = lr
        self.init_cost = init_cost
        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size    = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, attack_size=100,
                 init_m=None, init_p=None):
        source, target = pair

        cost = self.init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        mask_best    = torch.zeros(self.pattern_size).to(self.device)
        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float('inf')

        init_mask    = init_m if init_m is not None\
                            else np.random.random(self.mask_size)
        init_pattern = init_p if init_p is not None\
                            else np.random.random(self.pattern_size)
        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        self.mask_tensor    = torch.Tensor(init_mask).to(self.device)
        self.pattern_tensor = torch.Tensor(init_pattern).to(self.device)
        self.mask_tensor.requires_grad    = True
        self.pattern_tensor.requires_grad = True

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=self.lr, betas=(0.5, 0.9))

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                self.mask = (torch.tanh(self.mask_tensor) / (2 - self.epsilon)\
                                + 0.5).repeat(self.img_channels, 1, 1)
                self.pattern = (torch.tanh(self.pattern_tensor) /\
                                (2 - self.epsilon) + 0.5)

                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mask_best = self.mask
                pattern_best = self.pattern
                reg_best = avg_loss_reg

            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'.format(\
                            source, target, mask_best.abs().sum()))
        sys.stdout.flush()

        return mask_best, pattern_best


class DualTanh:
    def __init__(self, model, input_shape=(32, 32, 3), clip_max=1.0,
                 num_classes=10, batch_size=32, steps=1000, asr_bound=0.99,
                 lr=0.1, init_cost=1e-3, normalize=None, augment=False):
        self.model = model
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.clip_max = clip_max
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.asr_bound = asr_bound
        self.lr = lr
        self.init_cost = init_cost
        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(10),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.9, 1.0))
            ])

        self.device = torch.device('cuda')
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3):
        source, target = pair
        self.steps = steps

        cost = init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        pattern_best     = torch.zeros(self.pattern_size).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_size).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_size).to(self.device)
        reg_best = float('inf')
        l0_best  = float('inf')

        for i in range(2):
            init_pattern = np.random.random(self.pattern_size) * self.clip_max
            init_pattern = np.clip(init_pattern, 0.0, self.clip_max)
            init_pattern = init_pattern / self.clip_max

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(init_pattern).to(self.device)
                pattern_neg_tensor.requires_grad = True

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(y_set != target)[0]

            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [pattern_pos_tensor, pattern_neg_tensor],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        self.model.eval()

        index_base = np.arange(x_set.shape[0])
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                pattern_pos =   torch.clamp(pattern_pos_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)
                pattern_neg = - torch.clamp(pattern_neg_tensor * self.clip_max,
                                            min=0.0, max=self.clip_max)

                x_adv = torch.clamp(x_batch + pattern_pos + pattern_neg,
                                    min=0.0, max=self.clip_max)
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)

                acc = pred.eq(y_batch.view_as(pred)).sum().item()\
                            / pred.size(0)
                loss_ce = criterion(output, y_batch)

                mask_pos = torch.max(torch.tanh(pattern_pos_tensor / 10)\
                                / (2 - self.epsilon) + 0.5, axis=0)[0]
                mask_neg = torch.max(torch.tanh(pattern_neg_tensor / 10)\
                                / (2 - self.epsilon) + 0.5, axis=0)[0]
                loss_reg = torch.sum(mask_pos) + torch.sum(mask_neg)
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            if source == self.num_classes and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            threshold = self.clip_max / 255.0
            pattern_pos_cur = pattern_pos.detach()
            pattern_neg_cur = pattern_neg.detach()
            pattern_pos_cur[(pattern_pos_cur < threshold)\
                                & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold)\
                                & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur
            l0_cur = np.count_nonzero(np.sum(np.abs(pattern_cur.cpu().numpy()),
                                      axis=0))

            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best\
                    and l0_cur < l0_best:
                reg_best = avg_loss_reg
                l0_best = l0_cur

                pattern_pos_best = pattern_pos.detach()
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best / self.clip_max
                with torch.no_grad():
                    pattern_pos_tensor.copy_(init_pattern)

                pattern_neg_best = pattern_neg.detach()
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best / self.clip_max
                with torch.no_grad():
                    pattern_neg_tensor.copy_(init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best

                if source == self.num_classes\
                        and len(loss_ce_list) == attack_size:
                    loss_end[index_base] = loss_ce_list

            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best)\
                                 + 'l0: {:.0f}  '.format(l0_best))
                sys.stdout.flush()

        size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rl0 norm of pair {:d}-{:d}: {:d}\n'\
                         .format(source, target, size))
        sys.stdout.flush()

        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        return pattern_best, pattern_pos_best, pattern_neg_best, loss_monitor


class Blend:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, epsilon=0.06, a=0.01,
                 normalize=None, goal=False):
        self.model = model
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.epsilon = epsilon
        self.a = a
        self.normalize = normalize
        self.goal = goal

        self.device = torch.device('cuda')

        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        pattern_best = torch.zeros(self.pattern_size).to(self.device)
        acc_best = 0

        init_pattern = np.random.uniform(-self.epsilon, self.epsilon, self.pattern_size)
        self.pattern = torch.Tensor(init_pattern).to(self.device)
        self.pattern.requires_grad = True

        if self.goal:
            if source < self.num_classes:
                indices = np.where(y_set == source)[0]
            else:
                indices = np.where(y_set != target)[0]

            if indices.shape[0] > attack_size:
                indices = np.random.choice(indices, attack_size, replace=False)
            else:
                attack_size = indices.shape[0]
            x_set = x_set[indices].to(self.device)
            y_set = torch.full((x_set.shape[0],), target).to(self.device)
        else:
            x_set = x_set.to(self.device)
            y_set = y_set.to(sefl.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.pattern],
                                     lr=self.a, betas=(0.5, 0.9))

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            acc_list = []
            loss_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                x_adv = x_batch + torch.FloatTensor(x_batch.shape).uniform_(
                                    -self.epsilon, self.epsilon).to(self.device)
                x_adv = torch.clamp(x_adv, 0, 1)

                x_adv = x_adv + torch.clamp(self.pattern, -self.epsilon, self.epsilon)

                optimizer.zero_grad()
                output = self.model(self.normalize(x_adv))
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)
                if not self.goal:
                    acc = 1 - acc

                loss = criterion(output, y_batch).mean()

                loss.backward()
                optimizer.step()

                acc_list.append(acc)
                loss_list.append(loss.detach().cpu().numpy())

            acc_avg = np.mean(acc_list)

            if acc_avg >= acc_best:
                acc_best = acc_avg.copy()
                pattern_best = torch.clamp(self.pattern, -self.epsilon, self.epsilon)

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, best: {:.2f}'\
                                 .format(step, acc_avg, acc_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rpair {:d}-{:d}: asr {:.2f}, size {:.2f}\n'.format(\
                            source, target, acc_best, pattern_best.abs().max()))
        sys.stdout.flush()

        return pattern_best


class Filter:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, asr_bound=0.9, lr=0.01,
                 init_cost=1e-3, normalize=None, augment=False):
        self.model = model
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.asr_bound = asr_bound
        self.lr = lr
        self.init_cost = init_cost
        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3):
        source, target = pair
        self.steps = steps

        cost = init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        mean_best   = torch.zeros((self.img_channels)).to(self.device)
        std_best    = torch.zeros((self.img_channels)).to(self.device)
        acc_best = 0
        reg_best = float('inf')

        t_mean   = torch.empty((self.img_channels)).to(self.device)
        t_std    = torch.empty((self.img_channels)).to(self.device)
        nn.init.uniform_(t_mean)
        nn.init.uniform_(t_std)
        t_mean.requires_grad   = True
        t_std.requires_grad    = True

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(y_set != target)[0]

            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [t_mean, t_std],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        self.model.eval()
        index_base = np.arange(x_set.shape[0])
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                if self.augment:
                    x_batch = self.transform(x_batch)

                optimizer.zero_grad()

                x_mean = x_batch.mean((2, 3), keepdim=True)
                x_std  = x_batch.std( (2, 3), keepdim=True)
                mean_expand = t_mean.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                std_expand  = t_std.unsqueeze(1).unsqueeze(1).unsqueeze(0)

                x_adv = (x_batch - x_mean) / replacezero(x_std) * std_expand\
                                                                + mean_expand
                x_adv = torch.clamp(x_adv, min=0.0, max=1.0)

                y_adv = self.model(self.normalize(x_adv))
                pred = y_adv.argmax(dim=1, keepdim=True)

                acc = pred.eq(y_batch.view_as(pred)).sum().item()\
                                / pred.size(0)
                loss_ce  = criterion(y_adv, y_batch)

                loss_mean_reg = torch.sum(torch.abs(\
                                    t_mean - x_mean.mean(0, keepdim=True)))\
                                    / self.img_channels
                loss_std_reg  = torch.sum(torch.abs(\
                                    t_std  - x_std.mean( 0, keepdim=True)))\
                                    / self.img_channels
                loss_reg = loss_mean_reg + loss_std_reg

                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            if source == self.num_classes and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc  = np.mean(acc_list)

            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mean_best   = t_mean
                std_best    = t_std
                acc_best = avg_acc
                reg_best = avg_loss_reg

                if source == self.num_classes\
                        and len(loss_ce_list) == attack_size:
                    loss_end[index_base] = loss_ce_list

            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}'\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rreg loss of pair {:d}-{:d}: {:.2f}\n'\
                         .format(source, target, reg_best))
        sys.stdout.flush()

        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        reg_best = 0 if np.isinf(reg_best) else reg_best

        return mean_best, std_best, reg_best, loss_monitor


class DFST:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, asr_bound=0.9, lr=0.01,
                 init_cost=1e-3, normalize=None, augment=False):
        self.model = model
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.asr_bound = asr_bound
        self.lr = lr
        self.init_cost = init_cost
        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.kshape = [64, 64, 3, 3]
        self.bshape = [64]

        pretrain = M.vgg16(pretrained=True)
        self.encoder = torch.nn.Sequential(\
                                    *(list(pretrain.features.children())[:5]))
        self.encoder.to(self.device)
        self.encoder.eval()

        # imagenet: torch.pt, cifar: torch_9.pt
        self.decoder = torch.load('ckpt/imagenet_vgg16_decoder_torch_9.pt')
        self.decoder.to(self.device)
        self.decoder.eval()

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3):
        source, target = pair
        self.steps = steps

        cost = init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        mean_best   = torch.zeros((self.img_channels)).to(self.device)
        std_best    = torch.zeros((self.img_channels)).to(self.device)
        kernel_best = torch.zeros(self.kshape).to(self.device)
        bias_best   = torch.zeros(self.bshape).to(self.device)
        acc_best = 0
        reg_best = float('inf')
        con_best = float('inf')

        t_mean   = torch.empty((self.img_channels)).to(self.device)
        t_std    = torch.empty((self.img_channels)).to(self.device)
        t_kernel = torch.empty(self.kshape).to(self.device)
        t_bias   = torch.empty(self.bshape).to(self.device)
        nn.init.uniform_(t_mean)
        nn.init.uniform_(t_std)
        nn.init.xavier_normal_(t_kernel)
        nn.init.normal_(t_bias)
        t_mean.requires_grad   = True
        t_std.requires_grad    = True
        t_kernel.requires_grad = True
        t_bias.requires_grad   = True

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(y_set != target)[0]

            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        [t_kernel, t_bias, t_mean, t_std],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        self.model.eval()

        index_base = np.arange(x_set.shape[0])
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            index_base = index_base[indices]
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_con_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                if self.augment:
                    x_batch = self.transform(x_batch)

                optimizer.zero_grad()

                feature_ori = self.encoder(self.normalize(x_batch))

                x_mean = x_batch.mean((2, 3), keepdim=True)
                x_std  = x_batch.std( (2, 3), keepdim=True)
                mean_expand = t_mean.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                std_expand  = t_std.unsqueeze(1).unsqueeze(1).unsqueeze(0)

                x_norm = (x_batch - x_mean) / replacezero(x_std) * std_expand\
                                + mean_expand
                x_norm = torch.clamp(x_norm, min=0.0, max=1.0)

                feature_norm = self.encoder(self.normalize(x_norm))
                fori_mean = feature_norm.mean((2, 3), keepdim=True)
                fori_std  = feature_norm.std( (2, 3), keepdim=True)
                feature_norm = (feature_norm - fori_mean) / replacezero(fori_std)

                feature_adv = F.conv2d(feature_norm, t_kernel, t_bias, padding=1)
                feature_adv = F.dropout2d(feature_adv, p=0.1)

                x_adv = self.decoder(feature_adv)
                feature_new = self.encoder(self.normalize(x_adv))

                y_adv = self.model(self.normalize(x_adv))
                pred = y_adv.argmax(dim=1, keepdim=True)

                acc = pred.eq(y_batch.view_as(pred)).sum().item()\
                                / pred.size(0)
                loss_ce  = criterion(y_adv, y_batch)

                loss_mean_reg = torch.sum(torch.abs(\
                                    t_mean - x_mean.mean(0, keepdim=True)))\
                                    / self.img_channels
                loss_std_reg  = torch.sum(torch.abs(\
                                    t_std  - x_std.mean( 0, keepdim=True)))\
                                    / self.img_channels
                loss_norm = loss_mean_reg + loss_std_reg

                loss_content = F.mse_loss(feature_new, feature_ori)
                loss_ssim    = 1 - pytorch_msssim.ssim(x_batch, x_adv)
                loss_smooth  = F.mse_loss(x_adv,\
                                    F.avg_pool2d(x_adv, 3, stride=1, padding=1))

                loss_reg =   1e-3 * loss_content\
                           +  1e2 * loss_ssim\
                           + 5e-2 * loss_smooth\
                           +        loss_norm
                loss_con = 1 * loss_content

                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_con_list.append(loss_con.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            if source == self.num_classes and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss_con = np.mean(loss_con_list)
            avg_loss = np.mean(loss_list)
            avg_acc  = np.mean(acc_list)

            if avg_acc >= self.asr_bound and avg_loss_con < con_best:
                mean_best   = t_mean
                std_best    = t_std
                kernel_best = t_kernel
                bias_best   = t_bias
                acc_best = avg_acc
                reg_best = avg_loss_reg
                con_best = avg_loss_con

                if source == self.num_classes\
                        and len(loss_ce_list) == attack_size:
                    loss_end[index_base] = loss_ce_list

            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = self.init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}, '\
                                 .format(avg_loss_ce, avg_loss_reg, reg_best)\
                                 + 'con_best: {:.2f}'.format(con_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rreg loss of pair {:d}-{:d}: {:.2f}, {:.2f}\n'\
                         .format(source, target, reg_best, con_best))
        sys.stdout.flush()

        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        reg_best = 0 if np.isinf(reg_best) else reg_best
        con_best = 0 if np.isinf(con_best) else con_best

        return mean_best, std_best, kernel_best, bias_best, con_best, loss_monitor


class WaNet:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, kdim=12, lr=1e-2, cost=1e-3,
                 normalize=None, augment=False):
        self.model = model
        self.kdim = kdim
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        init_weights = torch.rand((1, 27, self.kdim, self.kdim)) * 1e-2
        init_bias    = torch.rand((1, 3, 1, 1)) * 1e-2
        weights = init_weights.to(self.device)
        bias    = init_bias.to(self.device)
        weights.requires_grad = True
        bias.requires_grad    = True

        asr_best     = 0
        weights_best = None
        bias_best    = None

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=[weights, bias],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                x_adv = wanet_trigger(x_batch, weights, bias)
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(weights))\
                                + torch.sum(torch.abs(bias))
                loss = loss_ce.mean() + loss_reg * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc > asr_best:
                asr_best     = avg_acc
                weights_best = weights
                bias_best    = bias

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return weights_best, bias_best


class Reflection:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, tp=0.7, tp_init=0.25,
                 lr=1e-1, cost=1e-1, normalize=None, augment=False):
        self.model = model
        self.tp = tp
        self.tp_init = tp_init
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        trigger_path = './data/reflection.jpg'
        trigger = read_image(trigger_path) / 255.0
        trigger = T.Resize((self.img_rows, self.img_cols))(trigger)

        noise = torch.rand((1, 1, self.img_rows, self.img_cols))
        noisy_trigger = self.tp * trigger + (1 - self.tp) * noise

        self.patn_init = noisy_trigger.clone().to(self.device)
        self.mask_init = torch.ones((1, 1, 1, 1)) * self.tp_init

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        mask = self.mask_init.clone().to(self.device)
        patn = self.patn_init.clone().to(self.device)
        mask.requires_grad = True
        patn.requires_grad = True

        asr_best  = 0
        mask_best = None
        patn_best = None

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=[mask, patn],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                tmask = torch.clamp(mask, 0., self.tp_init)
                x_adv = torch.clamp((1 - tmask) * x_batch + tmask * patn, 0., 1.)
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(tmask)\
                                + torch.sum((patn - self.patn_init) ** 2)
                loss = loss_ce.mean() + loss_reg * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc > asr_best:
                asr_best = avg_acc
                mask_best = mask
                patn_best = patn

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return mask_best, patn_best


class SIG:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, delta=50, frequency=4,
                 tp=0.8, tp_init=0.2, lr=1e-1, cost=1e-1,
                 normalize=None, augment=False):
        self.model = model
        self.delta = delta
        self.frequency = frequency
        self.tp = tp
        self.tp_init = tp_init
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        signal = np.zeros((1, 1, self.img_rows, self.img_cols))
        for i in range(self.img_rows):
            signal[:, :, :, i] += self.delta / 255.\
                                    * np.sin(2 * np.pi * i * self.frequency\
                                             / self.img_rows)
        trigger = torch.FloatTensor(signal)[0]

        noise = torch.rand((1, 1, self.img_rows, self.img_cols))
        noisy_trigger = self.tp * trigger + (1 - self.tp) * noise
        self.patn_init = noisy_trigger.clone().to(self.device)
        self.mask_init = torch.ones((1, 1, 1, 1)) * self.tp_init

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        mask = self.mask_init.clone().to(self.device)
        patn = self.patn_init.clone().to(self.device)
        mask.requires_grad = True
        patn.requires_grad = True

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=[mask, patn],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        asr_best  = 0
        mask_best = None
        patn_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                tmask = torch.clamp(mask, 0., self.tp_init)
                x_adv = torch.clamp((1 - tmask) * x_batch + tmask * patn, 0., 1.)
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(tmask)\
                                + torch.sum((patn - self.patn_init) ** 2)
                loss = loss_ce.mean() + loss_reg * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc > asr_best:
                asr_best = avg_acc
                mask_best = mask
                patn_best = patn

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return mask_best, patn_best


class Composite:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=32, steps=1000, lr=1e-1, cost=1e-3,
                 normalize=None, augment=False):
        self.model = model
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        self.stylegan = StyleGAN(self.device)

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        # Initialization of parameters
        mapping_labels\
                = nn.functional.one_hot(y_set, num_classes=self.num_classes)\
                                                    .float().to(self.device)
        latent_w_mean = self.stylegan.get_w_mean(mapping_labels)

        latent_input = latent_w_mean.clone()
        latent_input = latent_input.unsqueeze(1)\
                            .repeat(1, self.stylegan.num_layers, 1)
        latent_input.requires_grad_(True)

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=[latent_input],
                        lr=self.lr, betas=(0.5, 0.9)
                    )

        asr_best = 0
        gen_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]

                t = step / self.steps
                lr_i = get_lr(t, self.lr)
                optimizer.param_groups[0]['lr'] = lr_i

                special_noises = []
                gen = self.stylegan.generator(latent_input,
                                              special_noises=special_noises)
                x_adv = composite_trigger(x_batch, gen)
                x_adv = self.normalize(x_adv)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                loss_reg = TV_loss(gen)
                loss = loss_ce.mean() + loss_reg * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc > asr_best:
                asr_best = avg_acc
                gen_best = gen

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return gen_best


class GenL0:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=128, steps=75, lr=1e-3, cost=1, trigger_size=7,
                 normalize=None, augment=False):
        self.model = model
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        self.unet = UNet(n_channels=3,num_classes=3).cuda()
        self.trigger_size = trigger_size

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        self.unet.train()

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=self.unet.parameters(),
                        lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-4
                    )

        asr_best = 0
        gen_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = self.normalize(x_batch)

                t = step / self.steps
                lr_i = get_lr(t, self.lr)
                optimizer.param_groups[0]['lr'] = lr_i

                mask = torch.zeros((1,self.img_rows,self.img_cols)).cuda()
                mask[:,10:10 + self.trigger_size,10:10 + self.trigger_size] = 1

                trigger = self.unet(x_batch)
                trigger = nn.Tanh()(trigger)
                max_v = torch.FloatTensor([
                            (1 - self.normalize.mean[0]) / self.normalize.std[0],
                            (1 - self.normalize.mean[1]) / self.normalize.std[1],
                            (1 - self.normalize.mean[2]) / self.normalize.std[2]
                        ]).view(self.img_channels, 1, 1)\
                                .expand(self.img_channels,
                                        self.img_rows,
                                        self.img_cols).cuda()

                min_v = torch.FloatTensor([
                            (0 - self.normalize.mean[0]) / self.normalize.std[0],
                            (0 - self.normalize.mean[1]) / self.normalize.std[1],
                            (0 - self.normalize.mean[2]) / self.normalize.std[2]
                        ]).view(self.img_channels, 1, 1)\
                                .expand(self.img_channels,
                                        self.img_rows,
                                        self.img_cols).cuda()

                x_before_ae = x_batch
                trigger = trigger * 0.5 * (max_v - min_v) + 0.5 * (max_v + min_v)
                x_adv = (1 - mask) * x_batch + mask * trigger

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)
                bs = trigger.shape[0]
                half_bs = int(bs * 0.5)
                if half_bs * 2 > bs:
                    half_bs = half_bs - 1

                criterion_div = torch.nn.MSELoss(reduction='none').cuda()
                for iter_ in range(1):
                    index_1 = random.sample(range(bs), half_bs)
                    index_2 = random.sample(range(bs), half_bs)

                    distance_images = criterion_div(x_before_ae[index_1],
                                                    x_before_ae[index_2])
                    distance_images = torch.mean(distance_images, dim=(1, 2, 3))

                    distance_patterns\
                            = criterion_div(
                                    trigger[index_1, :, :self.trigger_size,\
                                                        :self.trigger_size],
                                    trigger[index_2, :, :self.trigger_size,\
                                                        :self.trigger_size])
                    distance_patterns = torch.mean(distance_patterns,
                                                   dim=(1, 2, 3))

                    loss_div = distance_images / (distance_patterns + 1e-6)
                    loss_div = torch.mean(loss_div) * 1

                    if iter_ == 0:
                        trigger_diverse_loss = loss_div
                    else:
                        trigger_diverse_loss = trigger_diverse_loss + loss_div

                loss = loss_ce.mean() + trigger_diverse_loss * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(self.trigger_size*self.trigger_size)
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= asr_best:
                asr_best = avg_acc
                gen_best = self.unet

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return gen_best


class GenL0Dynamic:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=128, steps=75, lr=1e-3, cost=1,
                 dataset='cifar10', trigger_size=7,
                 normalize=None, augment=False):
        self.model = model
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.dataset = dataset
        self.trigger_size = trigger_size

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        self.unet = UNet(n_channels=3,num_classes=3).cuda()

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        self.unet.train()

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=self.unet.parameters(),
                        lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-4
                    )

        asr_best = 0
        gen_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = self.normalize(x_batch)

                t = step / self.steps
                lr_i = get_lr(t, self.lr)
                optimizer.param_groups[0]['lr'] = lr_i

                x_before_ae = x_batch

                mask = torch.zeros((x_batch.shape[0], 1, self.img_rows,
                                   self.img_cols)).cuda()

                random_position_list = []

                if self.dataset == 'imagenet':
                    gap_x = int((size - self.trigger_size) / 128)
                    random_pos_x = random.randint(0, gap_x) * 128
                    gap_y = int((size - self.trigger_size) / 50)
                    random_pos_y = random.randint(0, gap_y) * 50
                if self.dataset == 'cifar10':
                    random_pos_x = random.randint(0, 3) * 3 + 10
                    random_pos_y = random.randint(0, 1) * 4 + 14

                for index in range(x_batch.shape[0]):
                    mask[index, :,\
                         random_pos_x : random_pos_x + self.trigger_size,\
                         random_pos_y : random_pos_y + self.trigger_size] = 1
                    random_position_list.append((random_pos_x, random_pos_y))

                trigger = self.unet(x_batch)
                trigger = nn.Tanh()(trigger)
                max_v = torch.FloatTensor([
                            (1 - self.normalize.mean[0]) / self.normalize.std[0],
                            (1 - self.normalize.mean[1]) / self.normalize.std[1],
                            (1 - self.normalize.mean[2]) / self.normalize.std[2]
                        ]).view(self.img_channels, 1, 1)\
                                .expand(self.img_channels,
                                        self.img_rows,
                                        self.img_cols).cuda()
                min_v = torch.FloatTensor([
                            (0 - self.normalize.mean[0]) / self.normalize.std[0],
                            (0 - self.normalize.mean[1]) / self.normalize.std[1],
                            (0 - self.normalize.mean[2]) / self.normalize.std[2]
                        ]).view(self.img_channels, 1, 1)\
                                .expand(self.img_channels,
                                        self.img_rows,
                                        self.img_cols).cuda()

                trigger = trigger * 0.5 * (max_v - min_v) + 0.5 * (max_v + min_v)
                x_adv = (1 - mask) * x_batch + mask * trigger

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce  = criterion(output, y_batch)

                bs = trigger.shape[0]
                half_bs = int(bs * 0.5)
                if half_bs * 2 > bs:
                    half_bs = half_bs - 1

                pure_trigger = torch.zeros((x_batch.shape[0], 3,
                                    self.trigger_size, self.trigger_size)).cuda()

                for index in range(x_batch.shape[0]):
                    pos_x = random_position_list[index][0]
                    pos_y = random_position_list[index][1]
                    pure_trigger[index, :, :, :]\
                            = trigger[index, :, pos_x:pos_x+self.trigger_size,\
                                                pos_y:pos_y+self.trigger_size]

                criterion_div = torch.nn.MSELoss(reduction='none').cuda()
                for iter_ in range(1):
                    index_1 = random.sample(range(bs),half_bs)
                    index_2 = random.sample(range(bs),half_bs)

                    distance_images = criterion_div(x_before_ae[index_1],
                                                    x_before_ae[index_2])
                    distance_images = torch.mean(distance_images, dim=(1, 2, 3))

                    distance_patterns = criterion_div(pure_trigger[index_1],
                                                      pure_trigger[index_2])
                    distance_patterns = torch.mean(distance_patterns,
                                                   dim=(1, 2, 3))

                    loss_div = distance_images / (distance_patterns + 1e-6)
                    loss_div = torch.mean(loss_div) * 1

                    if iter_ == 0:
                        trigger_diverse_loss = loss_div
                    else:
                        trigger_diverse_loss = trigger_diverse_loss + loss_div

                loss = loss_ce.mean() + trigger_diverse_loss * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(self.trigger_size*self.trigger_size)
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= asr_best:
                asr_best = avg_acc
                gen_best = self.unet

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return gen_best


class GenL2:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=128, steps=75, lr=1e-3, cost=100, l2_bound=0.1,
                 normalize=None, augment=False):
        self.model = model
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.l2_bound = l2_bound

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        self.unet = UNet(n_channels=3,num_classes=3).cuda()

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        self.unet.train()

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=self.unet.parameters(),
                        lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-4
                    )

        asr_best = 0
        gen_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = self.normalize(x_batch)

                t = step / self.steps
                lr_i = get_lr(t, self.lr)
                optimizer.param_groups[0]['lr'] = lr_i

                x_adv = self.unet(x_batch)
                loss_reg = nn.MSELoss(size_average = True).cuda()(x_adv, x_batch)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                loss_ce = criterion(output, y_batch)

                loss = loss_ce.mean()
                if loss_reg > self.l2_bound:
                    loss = loss + loss_reg * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= asr_best:
                asr_best = avg_acc
                gen_best = self.unet

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return gen_best

        
class InputAware:
    def __init__(self, model, input_shape=(32, 32, 3), num_classes=10,
                 batch_size=128, steps=75, lr=1e-3, cost=1, args=None,
                 normalize=None, augment=False):
        self.model = model
        self.lr = lr
        self.cost = cost

        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.img_channels = input_shape[2]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps

        self.normalize = normalize
        self.augment = augment

        if self.augment:
            self.transform = T.Compose([
                T.RandomRotation(1),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(self.img_rows, scale=(0.99, 1.0))
            ])

        self.device = torch.device('cuda')

        args.input_channel = input_shape[2]
        self.netG = Generator(args).cuda()
        self.netM = Generator(args, out_channels=1).cuda()

        state_dict = torch.load('ckpt/input_aware_netM.pth')
        self.netM.load_state_dict(state_dict)
        self.netM.eval()

    def generate(self, pair, x_set, y_set, attack_size=100):
        source, target = pair

        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set != target)[0]

        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].to(self.device)
        y_set = torch.full((x_set.shape[0],), target).to(self.device)

        if attack_size < self.batch_size:
            self.batch_size = attack_size

        self.netG.train()

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(
                        params=self.netG.parameters(),
                        lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-4
                    )

        asr_best = 0
        gen_best = None

        self.model.eval()
        for step in range(self.steps):
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = self.normalize(x_batch)

                t = step / self.steps
                lr_i = get_lr(t, self.lr)
                optimizer.param_groups[0]['lr'] = lr_i

                x_before_ae = x_batch

                patterns = self.netG(x_batch)
                patterns = self.netG.normalize_pattern(patterns)

                masks_output = self.netM.threshold(self.netM(x_batch))
                bd_inputs = x_batch + (patterns - x_batch) * masks_output
                x_adv = bd_inputs

                loss_reg = nn.MSELoss(size_average=True).cuda()(x_adv,x_batch)

                if self.augment:
                    x_adv = self.transform(x_adv)

                optimizer.zero_grad()

                output = self.model(x_adv)
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item() / pred.size(0)

                bs = x_batch.shape[0]
                half_bs = int(bs * 0.5)
                if half_bs * 2 > bs:
                    half_bs = half_bs - 1

                criterion_div = torch.nn.MSELoss(reduction='none').cuda()
                for iter_ in range(1):
                    index_1 = random.sample(range(bs),half_bs)
                    index_2 = random.sample(range(bs),half_bs)

                    distance_images = criterion_div(x_before_ae[index_1],
                                                    x_before_ae[index_2])
                    distance_images = torch.mean(distance_images, dim=(1, 2, 3))

                    distance_patterns = criterion_div(patterns[index_1, :, :, :],
                                                      patterns[index_2, :, :, :])
                    distance_patterns = torch.mean(distance_patterns,
                                                   dim=(1, 2, 3))

                    loss_div = distance_images / (distance_patterns + 1e-6)
                    loss_div = torch.mean(loss_div) * 1

                    if iter_ == 0:
                        trigger_diverse_loss = loss_div
                    else:
                        trigger_diverse_loss = trigger_diverse_loss + loss_div

                loss_ce  = criterion(output, y_batch)

                loss = loss_ce.mean() + trigger_diverse_loss * self.cost

                loss.backward()
                optimizer.step()

                loss_ce_list.extend(loss_ce.detach().cpu().numpy())
                loss_reg_list.append(0)
                loss_list.append(loss.detach().cpu().numpy())
                acc_list.append(acc)

            avg_loss_ce = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss = np.mean(loss_list)
            avg_acc = np.mean(acc_list)

            if avg_acc >= asr_best:
                asr_best = avg_acc
                gen_best = self.netG

            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                 .format(step, avg_acc, avg_loss)\
                                 + 'ce: {:.2f}, reg: {:.2f}, asr_best: {:.2f}  '\
                                 .format(avg_loss_ce, avg_loss_reg, asr_best))
                sys.stdout.flush()
        print()

        return self.netM, gen_best
