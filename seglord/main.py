import argparse
import random, torch
import numpy as np
import json

from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser(description="Segmentation Model CLI")

    # dataset
    parser.add_argument('--ds', type=str, required=True, help="dataset")
    parser.add_argument('--dt', type=str, help='data dir')
    parser.add_argument('--bs', type=int, required=True, help='batch size')
    parser.add_argument('--wk', type=str, default=8, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, nargs='+', required=False, help='size of processed image (h, w)')

    # model
    parser.add_argument('--model', type=str, choices=['unet', 'unetpp', 'manet', 'lnet', 'fpn', 'psp', 'pan', 'dl3', 'dl3p'], help='Model type')
    parser.add_argument('--ename', type=str, required=True, help='Encoder name')
    parser.add_argument('--edepth', type=int, required=True, help='Encoder depth')
    parser.add_argument('--eweight', type=str, default=None, choices=[None, 'imagenet'], help='Encoder weights')
    parser.add_argument('--bn', action='store_true', help='Use batch norm in decoder')
    parser.add_argument('--dchannels', type=int, required=True, help='Decoder channels')
    parser.add_argument('--att', type=str, help='Decoder attention type')
    parser.add_argument('--pab', type=int, help='Decoder PAB channels')
    parser.add_argument('--prm', type=int, help='Decoder pyramid channels')
    parser.add_argument('--segch', type=int, help='Decoder segmentation channels')
    parser.add_argument('--decmerge', type=str, help='Decoder merge policy')
    parser.add_argument('--ups', type=int, help='Upsampling factor')
    parser.add_argument('--pspoch', type=int, help='PSPNet output channels')
    parser.add_argument('--estride', type=int, help='Encoder output stride')
    parser.add_argument('--drates', type=int, nargs='+', help='Decoder atrous rates')

    # training
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--dvids', type=str, nargs='+', default=[0], help='devices')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # logging
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb')
    parser.add_argument('--wandb_prj', type=str, required=False, help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, required=False, help='wandb entity name')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False