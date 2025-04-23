import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import argparse
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms as T
from torchvision.io import read_image
from torchvision.utils import save_image

from inversion import NC, DualTanh, Composite, WaNet
from inversion import Blend, Reflection, SIG, Filter, DFST
from inversion import GenL0, GenL0Dynamic, GenL2, InputAware
from models.decoder import Inversion
from models.unet_model import UNet
from models.input_aware.models import Generator
from util import get_classes, get_loader, get_model, get_norm, get_size
from util import replacezero
from util import wanet_trigger, composite_trigger


def eval_acc(model, loader, device):
    model.eval()
    n_sample = 0
    n_correct = 0
    with torch.no_grad():
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()

    acc = n_correct / n_sample
    return acc


def test(args):
    print('Loading model...')
    model = get_model(args.dataset, args.network).to(args.device)
    model = torch.nn.DataParallel(model)
    model.eval()

    print('Testing...')
    test_loader = get_loader(args.dataset, False, args.batch_size)
    acc = eval_acc(model, test_loader, args.device)
    print(f'ACC: {acc:.4f}')


def validate(args):
    model = get_model(args.dataset, args.network).to(args.device)
    model = torch.nn.DataParallel(model)
    model.eval()

    num_classes = get_classes(args.dataset)
    source, target = list(map(int, args.pair.split('-')))

    input_shape = get_size(args.dataset)
    normalize, unnormalize = get_norm(args.dataset)

    prefix = f'{args.dataset}_{args.network}_{args.opt}_{source}-{target}'
    trigger_path = f'./data/trigger/{prefix}'
    image_path   = f'./data/images/{prefix}'

    if args.load:
        if args.opt in ['nc', 'reflection', 'sig']:
            mask    = np.load(f'{trigger_path}_mask.npy')
            pattern = np.load(f'{trigger_path}_pattern.npy')
            print('trigger size:', np.sum(np.abs(mask)))
            mask    = torch.Tensor(mask).to(args.device)
            pattern = torch.Tensor(pattern).to(args.device)
        elif args.opt in ['dualtanh', 'blend', 'composite']:
            pattern = np.load(f'{trigger_path}_pattern.npy')
            size = np.count_nonzero(np.sum(np.abs(pattern), axis=0))
            print('trigger size:', size)
            pattern = torch.Tensor(pattern).to(args.device)
        elif args.opt in ['patch', 'dynamic', 'invisible']:
            unet = UNet(n_channels=3,num_classes=3).cuda()
            state_dict = torch.load(f'{trigger_path}_unet.pth')
            unet.load_state_dict(state_dict)
            unet.eval()
        elif args.opt == 'input_aware':
            netG = Generator(args).cuda()
            state_dict = torch.load(f'{trigger_path}_netG.pth')
            netG.load_state_dict(state_dict)
            netG.eval()

            netM = Generator(args, out_channels=1).cuda()
            state_dict = torch.load('ckpt/input_aware_netM.pth')
            self.netM.load_state_dict(state_dict)
            netM.eval()
        elif args.opt == 'wanet':
            t_weights = np.load(f'{trigger_path}_weights.npy')
            t_bias    = np.load(f'{trigger_path}_bias.npy')
            t_weights = torch.Tensor(t_weights).to(args.device)
            t_bias    = torch.Tensor(t_bias).to(args.device)
        elif args.opt == 'filter':
            t_mean   = np.load(f'{trigger_path}_mean.npy').reshape(1, 3, 1, 1)
            t_std    = np.load(f'{trigger_path}_std.npy').reshape(1, 3, 1, 1)
            t_mean   = torch.Tensor(t_mean).to(args.device)
            t_std    = torch.Tensor(t_std).to(args.device)
        elif args.opt == 'dfst':
            inversion = DFST(model,
                             input_shape=input_shape,
                             num_classes=num_classes,
                             normalize=normalize)
            t_mean   = np.load(f'{trigger_path}_mean.npy').reshape(1, 3, 1, 1)
            t_std    = np.load(f'{trigger_path}_std.npy').reshape(1, 3, 1, 1)
            t_kernel = np.load(f'{trigger_path}_kernel.npy')
            t_bias   = np.load(f'{trigger_path}_bias.npy')
            t_mean   = torch.Tensor(t_mean).to(args.device)
            t_std    = torch.Tensor(t_std).to(args.device)
            t_kernel = torch.Tensor(t_kernel).to(args.device)
            t_bias   = torch.Tensor(t_bias).to(args.device)
    else:
        print('Loading data...')
        train_loader = get_loader(args.dataset, True, args.batch_size)
        for i, (x_batch, y_batch) in enumerate(train_loader):
            if source < num_classes:
                indices = np.where(y_batch == source)[0]
            else:
                indices = np.where(y_batch != target)[0]

            if i == 0:
                x_val, y_val = x_batch[indices], y_batch[indices]
            else:
                x_val = torch.cat((x_val, x_batch[indices]))
                y_val = torch.cat((y_val, y_batch[indices]))

            if i > 8 and x_val.size(0) >= args.attack_size:
                break

            sys.stdout.write(f'\r{i}: {x_val.size(0)}')

        x_val = unnormalize(x_val)

        if args.opt == 'nc':
            inversion = NC(model,
                           input_shape=input_shape,
                           num_classes=num_classes,
                           asr_bound=args.asr_bound,
                           normalize=normalize)
            mask, pattern = inversion.generate((source, target), x_val, y_val,
                                               attack_size=args.attack_size)

            np.save(f'{trigger_path}_mask',    mask.detach().cpu().numpy())
            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(mask * pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'dualtanh':
            inversion = DualTanh(model,
                                 input_shape=input_shape,
                                 num_classes=num_classes,
                                 asr_bound=args.asr_bound,
                                 normalize=normalize)

            (pattern, _, _, _)\
                    = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'patch':
            inversion = GenL0(model,
                              input_shape=input_shape,
                              num_classes=num_classes,
                              cost=1,
                              normalize=normalize,
                              trigger_size=args.trigger_size)

            unet = inversion.generate((source, target), x_val, y_val,
                                      attack_size=args.attack_size)
            unet.eval()
            torch.save(unet.state_dict(), f'{trigger_path}_unet.pth')
        elif args.opt == 'dynamic':
            inversion = GenL0Dynamic(model,
                                     input_shape=input_shape,
                                     num_classes=num_classes,
                                     cost=1,
                                     normalize=normalize,
                                     dataset=args.dataset,
                                     trigger_size=args.trigger_size)

            unet = inversion.generate((source, target), x_val, y_val,
                                      attack_size=args.attack_size)
            unet.eval()
            torch.save(unet.state_dict(), f'{trigger_path}_unet.pth')
        elif args.opt == 'input_aware':
            inversion = InputAware(model,
                                   input_shape=input_shape,
                                   num_classes=num_classes,
                                   cost=1,
                                   normalize=normalize,
                                   args=args)

            netM, netG = inversion.generate((source, target), x_val, y_val,
                                            attack_size=args.attack_size)
            netM.eval()
            netG.eval()
            torch.save(netG.state_dict(), f'{trigger_path}_netG.pth')
        elif args.opt == 'composite':
            inversion = Composite(model,
                                  input_shape=input_shape,
                                  num_classes=num_classes,
                                  normalize=normalize)

            pattern = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'wanet':
            inversion = WaNet(model,
                              input_shape=input_shape,
                              num_classes=num_classes,
                              normalize=normalize)

            t_weights, t_bias\
                    = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_weights', t_weights.detach().cpu().numpy())
            np.save(f'{trigger_path}_bias',    t_bias.detach().cpu().numpy())
        elif args.opt == 'invisible':
            inversion = GenL2(model,
                              input_shape=input_shape,
                              num_classes=num_classes,
                              cost=100,
                              normalize=normalize,
                              l2_bound=args.l2_bound)

            unet = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)
            unet.eval()
            torch.save(unet.state_dict(), f'{trigger_path}_unet.pth')
        elif args.opt == 'blend':
            inversion = Blend(model,
                              input_shape=input_shape,
                              num_classes=num_classes,
                              normalize=normalize,
                              goal=True)
            pattern = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'reflection':
            inversion = Reflection(model,
                                   input_shape=input_shape,
                                   num_classes=num_classes,
                                   normalize=normalize)
            mask, pattern = inversion.generate((source, target), x_val, y_val,
                                               attack_size=args.attack_size)

            np.save(f'{trigger_path}_mask',    mask.detach().cpu().numpy())
            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(mask * pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'sig':
            inversion = SIG(model,
                            input_shape=input_shape,
                            num_classes=num_classes,
                            normalize=normalize)
            mask, pattern = inversion.generate((source, target), x_val, y_val,
                                               attack_size=args.attack_size)

            np.save(f'{trigger_path}_mask',    mask.detach().cpu().numpy())
            np.save(f'{trigger_path}_pattern', pattern.detach().cpu().numpy())
            save_image(mask * pattern, f'{trigger_path}_trigger.png')
        elif args.opt == 'filter':
            inversion = Filter(model,
                               input_shape=input_shape,
                               num_classes=num_classes,
                               asr_bound=args.asr_bound,
                               normalize=normalize)
            (t_mean, t_std, t_reg, _)\
                    = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_mean',   t_mean.detach().cpu().numpy())
            np.save(f'{trigger_path}_std',    t_std.detach().cpu().numpy())

            t_mean = t_mean.reshape(1, 3, 1, 1)
            t_std  = t_std.reshape(1, 3, 1, 1)
        elif args.opt == 'dfst':
            inversion = DFST(model,
                             input_shape=input_shape,
                             num_classes=num_classes,
                             asr_bound=args.asr_bound,
                             normalize=normalize)
            (t_mean, t_std, t_kernel, t_bias, t_reg, _)\
                    = inversion.generate((source, target), x_val, y_val,
                                         attack_size=args.attack_size)

            np.save(f'{trigger_path}_mean',   t_mean.detach().cpu().numpy())
            np.save(f'{trigger_path}_std',    t_std.detach().cpu().numpy())
            np.save(f'{trigger_path}_kernel', t_kernel.detach().cpu().numpy())
            np.save(f'{trigger_path}_bias',   t_bias.detach().cpu().numpy())

            t_mean = t_mean.reshape(1, 3, 1, 1)
            t_std  = t_std.reshape(1, 3, 1, 1)

    print('Validating trigger...')
    test_loader = get_loader(args.dataset, False, args.batch_size)
    n_sample = 0
    n_correct = 0
    saved = False
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            sys.stdout.write(f'\r{i}/{len(test_loader)}')
            if source < num_classes:
                indices = np.where(y_batch == source)[0]
            else:
                indices = np.where(y_batch != target)[0]
            if indices.shape[0] == 0:
                continue

            x_batch = x_batch[indices].to(args.device)
            y_batch = torch.full((x_batch.shape[0],), target).to(args.device)

            x_batch = unnormalize(x_batch)

            if args.opt in ['nc', 'reflection', 'sig']:
                x_batch = torch.clamp((1 - mask) * x_batch + mask * pattern, 0, 1)
            elif args.opt in ['dualtanh', 'blend']:
                x_batch = torch.clamp(x_batch + pattern, 0, 1)
            elif args.opt == 'patch':
                x_batch = normalize(x_batch)
                trigger = unet(x_batch)
                trigger = nn.Tanh()(trigger)
                img_channels = input_shape[2]
                img_rows = input_shape[0]
                img_cols = input_shape[1]

                max_v = torch.FloatTensor([
                            (1 - normalize.mean[0]) / normalize.std[0],
                            (1 - normalize.mean[1]) / normalize.std[1],
                            (1 - normalize.mean[2]) / normalize.std[2]
                        ]).view(img_channels, 1, 1)\
                                .expand(img_channels, img_rows, img_cols).cuda()
                min_v = torch.FloatTensor([
                            (0 - normalize.mean[0]) / normalize.std[0],
                            (0 - normalize.mean[1]) / normalize.std[1],
                            (0 - normalize.mean[2]) / normalize.std[2]
                        ]).view(img_channels, 1, 1)\
                                .expand(img_channels, img_rows, img_cols).cuda()
                mask = torch.zeros((1, img_rows, img_cols)).cuda()
                mask[:, 10:10+args.trigger_size, 10:10+args.trigger_size] = 1
                trigger = trigger * 0.5 * (max_v - min_v) + 0.5 * (max_v + min_v)
                x_batch = (1 - mask) * x_batch + mask * trigger
                x_batch = unnormalize(x_batch)
            elif args.opt == 'dynamic':
                x_batch = normalize(x_batch)
                trigger = unet(x_batch)
                trigger = nn.Tanh()(trigger)

                img_channels = input_shape[2]
                img_rows = input_shape[0]
                img_cols = input_shape[1]
                mask = torch.zeros((x_batch.shape[0], 1, img_rows, img_cols))\
                                                                        .cuda()

                for index in range(x_batch.shape[0]):
                    if args.dataset == 'imagenet':
                        gap_x = int((size - args.trigger_size) / 128)
                        random_pos_x = random.randint(0, gap_x) * 128
                        gap_y = int((size - args.trigger_size) / 50)
                        random_pos_y = random.randint(0, gap_y) * 50
                    if args.dataset == 'cifar10':
                        random_pos_x = random.randint(0, 3) * 3 + 10
                        random_pos_y = random.randint(0, 1) * 4 + 14

                    mask[index, :, random_pos_x:random_pos_x+args.trigger_size,\
                            random_pos_y:random_pos_y+args.trigger_size] = 1

                max_v = torch.FloatTensor([
                            (1 - normalize.mean[0]) / normalize.std[0],
                            (1 - normalize.mean[1]) / normalize.std[1],
                            (1 - normalize.mean[2]) / normalize.std[2]
                        ]).view(img_channels, 1, 1)\
                                .expand(img_channels, img_rows, img_cols).cuda()
                min_v = torch.FloatTensor([
                            (0 - normalize.mean[0]) / normalize.std[0],
                            (0 - normalize.mean[1]) / normalize.std[1],
                            (0 - normalize.mean[2]) / normalize.std[2]
                        ]).view(img_channels, 1, 1)\
                                .expand(img_channels, img_rows, img_cols).cuda()
                trigger = trigger * 0.5 * (max_v - min_v) + 0.5 * (max_v + min_v)
                x_batch = (1 - mask) * x_batch + mask * trigger
                x_batch = unnormalize(x_batch)
            elif args.opt == 'input_aware':
                x_batch = normalize(x_batch)
                patterns = netG(x_batch)
                patterns = netG.normalize_pattern(patterns)

                masks_output = netM.threshold(netM(x_batch))
                bd_inputs = x_batch + (patterns - x_batch) * masks_output
                x_batch = bd_inputs
                x_batch = unnormalize(x_batch)
            elif args.opt == 'composite':
                x_batch = composite_trigger(x_batch, pattern)
            elif args.opt == 'wanet':
                x_batch = wanet_trigger(x_batch, t_weights, t_bias)
            elif args.opt == 'filter':
                x_mean = x_batch.mean((2, 3), keepdim=True)
                x_std  = x_batch.std( (2, 3), keepdim=True)

                x_adv = (x_batch - x_mean) / replacezero(x_std) * t_std + t_mean
                x_batch = torch.clip(x_adv, 0.0, 1.0)
            elif args.opt == 'dfst':
                x_mean = x_batch.mean((2, 3), keepdim=True)
                x_std  = x_batch.std( (2, 3), keepdim=True)

                x_adv = (x_batch - x_mean) / replacezero(x_std) * t_std + t_mean
                x_adv = torch.clip(x_adv, 0.0, 1.0)

                feature_adv = inversion.encoder(normalize(x_adv))
                fori_mean = feature_adv.mean((2, 3), keepdim=True)
                fori_std  = feature_adv.std( (2, 3), keepdim=True)
                feature_adv = (feature_adv - fori_mean) / replacezero(fori_std)

                feature_adv = F.conv2d(feature_adv, t_kernel, t_bias, padding=1)
                x_batch = inversion.decoder(feature_adv)
            elif args.opt == 'invisible':
                x_batch = unet(normalize(x_batch))
                x_batch = unnormalize(x_batch)

            if not saved:
                for j in range(min(10, x_batch.shape[0])):
                    save_image(x_batch[j], f'{image_path}_{j}.png')
                saved = True

            output = model(normalize(x_batch))
            pred = output.max(dim=1)[1]

            n_sample  += x_batch.size(0)
            n_correct += (pred == y_batch).sum().item()
    asr = n_correct / n_sample
    print()
    print(f'Num: {n_sample}')
    print(f'ASR: {asr:.4f}')



###############################################################################
############                          main                         ############
###############################################################################
def main(args):
    if args.phase == 'test':
        test(args)
    elif args.phase == 'validate':
        validate(args)
    else:
        print('Option [{}] is not supported!'.format(args.phase))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')

    parser.add_argument('--gpu',     default='0',        help='gpu id')
    parser.add_argument('--phase',   default='validate', help='phase of framework')
    parser.add_argument('--dataset', default='cifar10',  help='dataset')
    parser.add_argument('--network', default='resnet20', help='model structure')
    parser.add_argument('--opt',     default='nc',       help='trigger optimization')
    parser.add_argument('--pair',    default='0-1',      help='label pair')

    parser.add_argument('--seed',        default=1024, type=int,   help='random seed')
    parser.add_argument('--batch_size',  default=128,  type=int,   help='batch size')
    parser.add_argument('--attack_size', default=100,  type=int,   help='attack size')
    parser.add_argument('--asr_bound',   default=0.99, type=float, help='asr bound for inversion')
    parser.add_argument('--l2_bound',    default=0.1,  type=float, help='l2 bound for invisible')
    parser.add_argument('--trigger_size',default=7,    type=int,   help='trigger size for patch and dynamic')

    parser.add_argument('--load', action='store_true', help='load generated trigger')

    args = parser.parse_args()
    args.device = torch.device('cuda')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_start = time.time()
    main(args)
    time_end = time.time()
    print('='*50)
    print('Running time:', (time_end - time_start) / 60, 'm')
    print('='*50)
