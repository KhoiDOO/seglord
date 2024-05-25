from accelerate import Accelerator

from data import get_data
from models import get_model
from losses import get_loss

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import random, torch
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Segmentation Model CLI")

    # dataset
    parser.add_argument('--ds', type=str, default='citynormal', help="dataset", choices=['citynormal'])
    parser.add_argument('--dt', type=str, help='data dir')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--wk', type=str, default=8, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, nargs='+', required=False, help='size of processed image (h, w)')

    # model
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unetpp', 'manet', 'lnet', 'fpn', 'psp', 'pan', 'dl3', 'dl3p'], help='Model type')
    parser.add_argument('--ename', type=str, default='resnet18', help='Encoder name')
    parser.add_argument('--edepth', type=int, default=5, help='Encoder depth', choices=[3, 4, 5])
    parser.add_argument('--eweight', type=str, default=None, help='Encoder weights', choices=[None, 'imagenet'])
    parser.add_argument('--bn', action='store_true', help='Use batch norm in decoder')
    parser.add_argument('--dchannels', type=int, nargs='+', default=[256, 128, 64, 32, 16], help='Decoder channels')
    parser.add_argument('--dl3dchannels', type=int, default=256, help='Deeplab v3 decoder channel')
    parser.add_argument('--att', type=str, default=None, help='Decoder attention type')
    parser.add_argument('--pab', type=int, default=64, help='Decoder PAB channels')
    parser.add_argument('--prm', type=int, default=256, help='Decoder pyramid channels')
    parser.add_argument('--segch', type=int, default=128, help='Decoder segmentation channels')
    parser.add_argument('--decmerge', type=str, default='add', help='Decoder merge policy', choices=['add', 'cat'])
    parser.add_argument('--ups', type=int, default=4, help='Upsampling factor', choices=[4, 8])
    parser.add_argument('--pspoch', type=int, default=512, help='PSPNet output channels')
    parser.add_argument('--estride', type=int, help='Encoder output stride')
    parser.add_argument('--drates', type=int, nargs='+', default=[12, 24, 36], help='Decoder atrous rates')

    # training
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--loss', type=str, default='ce', help='loss function', choices=['ce', 'dice', 'logdice', 'jaccard', 'logjaccard'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # logging
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb')
    parser.add_argument('--wandb_prj', type=str, default='seglord', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='truelove', help='wandb entity name')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Data Loaders
    train_ld, valid_ld, args = get_data(args=args)

    # Model
    model = get_model(args=args)
    criterion = get_loss(args=args)
    opt = Adam(model.parameters(), lr=args.lr)
    lrd = CosineAnnealingLR(optimizer=opt, T_max=len(train_ld))

    # Wandb

    # Accelerator Preparation
    ddp_train_ld, ddp_valid_ld, ddp_model, ddp_opt, ddp_lrd = accelerator.prepare(train_ld, valid_ld, model, opt, lrd)

    # Training
    


if __name__ == '__main__':
    main()