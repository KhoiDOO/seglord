from accelerate import Accelerator
from accelerate.utils import set_seed

from data import get_data
from models import get_model
from losses import get_loss
from metrics import Evaluator

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Module
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import *
from time import time

import argparse
import torch

TQDM_NCOLS = 50
TQDM_COLOUR = 'MAGENTA'


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
    parser.add_argument('--verbose', type=int, default=1, help='logging status')
    parser.add_argument('--me', type=str, default='miou', help='metric used for model saving', choices=['loss', 'miou', 'dice', 'dice', 'acc', 'f1', 'auc', 'roc', 'sen', 'spe'])

    # logging
    parser.add_argument('--wandb', action='store_true', help='toggle to use wandb')
    parser.add_argument('--wandb_prj', type=str, default='seglord', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='truelove', help='wandb entity name')
    parser.add_argument('--cache', action='store_true', help='cache logging info before syncing')

    args = parser.parse_args()
    set_seed(args.seed, deterministic=True)

    # Run
    args = folder_setup(args=args)

    # Data Loaders
    train_ld, valid_ld, args = get_data(args=args)

    # Model
    model = get_model(args=args)
    criterion = get_loss(args=args)
    opt = Adam(model.parameters(), lr=args.lr)
    lrd = CosineAnnealingLR(optimizer=opt, T_max=args.epoch * len(train_ld))

    pcnt = param_cnt(model=model)
    print(f"Total number of Params: {pcnt}")
    args.pcnt = pcnt

    # Accelerator
    accelerator = Accelerator(log_with="wandb" if args.wandb else None)
    device = accelerator.device
    print(f"Device in use: {device}")
    if args.wandb:
        accelerator.init_trackers(project_name=args.wandb_prj, config=vars(args), 
                                  init_kwargs={'entity': args.wandb_entity, 'name' : args.runid, 'force' : True})

    # Accelerator Preparation

    ddp_train_ld: DataLoader = accelerator.prepare_data_loader(train_ld)
    ddp_valid_ld: DataLoader = accelerator.prepare_data_loader(valid_ld)
    ddp_model: Module = accelerator.prepare_model(model)
    ddp_opt: Optimizer = accelerator.prepare_optimizer(opt)
    ddp_lrd: CosineAnnealingLR = accelerator.prepare_scheduler(lrd)
    # ddp_model, ddp_opt, ddp_lrd, ddp_train_ld, ddp_valid_ld = accelerator.prepare(model, opt, lrd, train_ld, valid_ld)

    # Logging
    evaluator = Evaluator(args=args)

    # Training
    with tqdm(total=args.epochs, ncols=TQDM_NCOLS, colour=TQDM_COLOUR, disable=False if accelerator.is_main_process else True) as pbar:

        for epoch in range(args.epochs):
            
            train_start = time()
            ddp_model.train()
            for (x, y_true) in ddp_train_ld:
                
                ddp_opt.zero_grad()
                
                y_pred = ddp_model(x)
                loss = evaluator(criterion=criterion, y_pred=y_pred, y_true=y_true, mode='train')
                
                accelerator.backward(loss)

                ddp_opt.step()
                ddp_lrd.step()
            train_time = time() - train_start
            
            valid_start = time()
            ddp_model.eval()
            with torch.no_grad():
                for (x, y_true) in ddp_valid_ld:
                    
                    y_pred = ddp_model(x)
                    evaluator(criterion=criterion, y_pred=y_pred, y_true=y_true, mode='valid')
            valid_time = time() - valid_start

            evaluator.add(value=train_time, key='train/time')
            evaluator.add(value=valid_time, key='valid/time')
            
            pbar.set_description(f'Epoch: {epoch} - {evaluator.txt}')
            pbar.update(1)
            evaluator.step(accelerator=accelerator, model=ddp_model)
    
    evaluator.sync(accelerator=accelerator)
    accelerator.end_training()
        

if __name__ == '__main__':
    main()