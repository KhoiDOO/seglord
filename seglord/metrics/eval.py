from accelerate import Accelerator
from accelerate.tracking import WandBTracker

from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F

from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from torchmetrics.classification import \
    Accuracy, BinaryAccuracy, \
    F1Score, BinaryF1Score, \
    AUROC, BinaryAUROC, \
    ROC, BinaryROC, \
    Recall, BinaryRecall, \
    Specificity, BinarySpecificity

from einops import rearrange

from typing import List
from .utils import *
from glob import glob
from tqdm import tqdm

import argparse
import torch
import wandb
import warnings

TQDM_NCOLS = None
TQDM_COLOUR = 'MAGENTA'


class Evaluator:
    def __init__(self, args: argparse) -> None:
        self.__log_dict = {}

        self.cls = args.cls
        self.trldcnt = args.trldcnt
        self.vlldcnt = args.vlldcnt
        self.vb = args.verbose
        self.wb = args.wandb
        self.runid = args.runid
        self.svdir = args.svdir
        self.me = args.me
        self.cache = args.cache
        self.args = args

        self.old_metric_value = 1e26 if self.me in ['loss'] else 0
        self.cpfunc = self.compare_func()
        self.best_path = self.svdir + "/best.pt"
        self.last_path = self.svdir + "/last.pt"

    def __call__(self, criterion: Module, y_pred: Tensor, y_true: Tensor, mode: str) -> Tensor:
        """_summary_

        Args:
            criterion (Module): loss function module
            y_pred (Tensor): prediction with shape (B, C, H, W)
            y_true (Tensor): ground-truth with shape (B, C, H, W)
            mode (str): 'train' or 'valid

        Returns:
            Tensor: loss for backward
        """

        bcnt = self.trldcnt if mode == 'train' else self.vlldcnt
        task = 'multiclass' if self.cls != 1 else 'binary'
        device = y_pred.device

        y_pred_idx = torch.argmax(y_pred, dim=1) if task == 'multiclass' else torch.sigmoid(y_pred)
        y_true_idx = torch.argmax(y_true, dim=1) if task == 'multiclass' else y_true

        _y_pred_flat = rearrange(y_pred, 'b c h w -> b h w c').flatten(0, 2)
        _y_true_flat = rearrange(y_true, 'b c h w -> b h w c').flatten(0, 2)

        _y_pred = torch.argmax(_y_pred_flat, dim=1) if task == 'multiclass' else torch.sigmoid(_y_pred_flat)
        _y_true = torch.argmax(_y_true_flat, dim=1) if task == 'multiclass' else _y_true_flat
        
        # loss
        loss = criterion(y_pred, y_true)
        self.update(value=loss, mode=mode, metric_name='loss', bcnt=bcnt)

        # miou
        miou = MeanIoU(num_classes=self.cls).to(device=device)(y_pred_idx, y_true_idx)
        self.update(value=miou, mode=mode, metric_name='miou', bcnt=bcnt)
        
        # dice
        dice = GeneralizedDiceScore(num_classes=self.cls).to(device=device)(y_pred_idx, y_true_idx)
        self.update(value=dice, mode=mode, metric_name='dice', bcnt=bcnt)

        # acc
        acc = Accuracy(task=task, num_classes=self.cls).to(device=device)(_y_pred, _y_true)
        self.update(value=acc, mode=mode, metric_name='acc', bcnt=bcnt)

        # f1
        f1 = F1Score(task=task, num_classes=self.cls).to(device=device)(_y_pred, _y_true)
        self.update(value=f1, mode=mode, metric_name='f1', bcnt=bcnt)

        # auc
        # auc = AUROC(task=task, num_classes=self.cls).to(device=device)(_y_pred_flat, _y_true)
        # self.update(value=auc, mode=mode, metric_name='auc', bcnt=bcnt)

        # roc
        # roc = ROC(task=task, num_classes=self.cls).to(device=device)(_y_pred, _y_true)
        # self.update(value=roc, mode=mode, metric_name='roc', bcnt=bcnt)

        # sen
        sens = Recall(task=task, num_classes=self.cls).to(device=device)(_y_pred, _y_true)
        self.update(value=sens, mode=mode, metric_name='sen', bcnt=bcnt)

        # spe
        spec = Specificity(task=task, num_classes=self.cls).to(device=device)(_y_pred, _y_true)
        self.update(value=spec, mode=mode, metric_name='spe', bcnt=bcnt)

        if task == 'multiclass':

            _y_pred_keep = F.one_hot(torch.argmax(_y_pred_flat, dim=1))

            # clswise miou
            mious = MeanIoU(num_classes=self.cls, per_class=True).to(device=device)(y_pred_idx, y_true_idx)
            self.clswise_update(values=mious, mode=mode, metric_name='miou', bcnt=bcnt)

            # clswise dice
            dices = GeneralizedDiceScore(num_classes=self.cls, per_class=True).to(device=device)(y_pred_idx, y_true_idx)
            self.clswise_update(values=dices, mode=mode, metric_name='dice', bcnt=bcnt)

            # clswise spe
            self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinarySpecificity().to(device=device), mode=mode, metric_name='spe', bcnt=bcnt)

            # clswise acc
            self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinaryAccuracy().to(device=device), mode=mode, metric_name='acc', bcnt=bcnt)

            # clswise f1 
            self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinaryF1Score().to(device=device), mode=mode, metric_name='f1', bcnt=bcnt)

            # clswise sens 
            self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinaryRecall().to(device=device), mode=mode, metric_name='sen', bcnt=bcnt)

            # clswise auc 
            # self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinaryAUROC().to(device=device), mode=mode, metric_name='auc', bcnt=bcnt)

            # clswise roc 
            # self.clswise_update_metric(y_pred=_y_pred_keep, y_true=_y_true_flat, fn=BinaryROC().to(device=device), mode=mode, metric_name='roc', bcnt=bcnt)

        return loss

    @property
    def log(self):
        return self.__log_dict

    @property
    def txt(self):
        if self.vb == 0:
            return 'non verbose'
        elif self.vb == 1:
            return " - ".join([f'{key}: {value:.2f}' for key, value in self.__log_dict.items() if key.split('/')[-1] in ['loss', 'miou', 'dice']])
        # elif self.vb == 2:
        #     return " - ".join([f'{key}: {value:.2f}' for key, value in self.__log_dict.items()])
    
    def step(self, accelerator: Accelerator, model: Module | DistributedDataParallel, epoch: int):

        if self.wb:
            if self.cache:
                save_json(dct=self.__log_dict, path=self.svdir + f'/log_{epoch}.json')
            else:
                accelerator.log(self.__log_dict)
        else:
            save_json(dct=self.__log_dict, path=self.svdir + f'/log_{epoch}.json')
        
        unwrap_model: Module = accelerator.unwrap_model(model)

        save_dict = {
            'args' : self.args,
            'model_state_dict': unwrap_model.state_dict()
        }

        metric_value = self.__log_dict[f'valid/{self.me}']
        
        if self.cpfunc(metric_value, self.old_metric_value):
            torch.save(save_dict, self.best_path)
            self.old_metric_value = metric_value
        
        torch.save(save_dict, self.last_path)
        
        self.__log_dict = {}
    
    def sync(self, accelerator: Accelerator):
        
        if self.cache:
            files = glob(self.svdir + '/*.json')

            with tqdm(total=len(files), ncols=TQDM_NCOLS, colour=TQDM_COLOUR) as pbar:
                pbar.set_description('Syncing')
                for file in files:
                    dct = read_json(path=file)
                    accelerator.log(values=dct)
                    pbar.update(1)
        if self.wb and accelerator.is_main_process:
            wandb_tracker: WandBTracker = accelerator.get_tracker('wandb')

            best_model_art = wandb.Artifact(name=f'{self.runid}-best', type='model')
            best_model_art.add_file(local_path=self.best_path)

            last_model_art = wandb.Artifact(name=f'{self.runid}-last', type='model')
            last_model_art.add_file(local_path=self.last_path)

            wandb_tracker.run.log_artifact(best_model_art)
            wandb_tracker.run.log_artifact(last_model_art)

    def clswise_update(self, values: Tensor | List, mode: str, metric_name: str, bcnt: int):
        
        values = values.tolist() if isinstance(values, Tensor) else values
        
        for idx, value in enumerate(values):
            key = f'{mode}/{idx}-{metric_name}'

            value = value.item() if isinstance(value, Tensor) else value

            if key in self.__log_dict:
                self.__log_dict[key] += value / bcnt
            else:
                self.__log_dict[key] = value / bcnt
    
    def clswise_update_metric(self, y_pred: Tensor, y_true: Tensor, fn: Module, mode: str, metric_name: str, bcnt: int):
        """_summary_

        Args:
            y_pred (Tensor): prediction with shape (B * H * W, C)
            y_true (Tensor): ground-truth with shape (B * H * W, C)
            fn (Module): metric function
            mode (str): 'train' or 'valid
            metric_name (str): name of metric
            bcnt (int): number of batchss
        """

        values = [fn(y_pred[:, idx], y_true[:, idx]) for idx in range(self.cls)]

        self.clswise_update(values=values, mode=mode, metric_name=metric_name, bcnt=bcnt)
    
    def update(self, value: Tensor | int, mode: str, metric_name: str, bcnt: int):
        
        key = f'{mode}/{metric_name}'
        
        value = value.item() if isinstance(value, Tensor) else value
        
        if key in self.__log_dict:
            self.__log_dict[key] += value / bcnt
        else:
            self.__log_dict[key] = value / bcnt
    
    def compare_func(self):
        def bigger(cur: float, old: float):
            return cur >= old
        def smaller(cur: float, old: float):
            return cur <= old
        
        return smaller if self.args.me in ['loss'] else bigger

    def add(self, value: int | float, key: str):
        if key in self.__log_dict:
            old_value = self.__log_dict[key]
            warnings.warn(f'There exist {key} in the database, the old value {old_value} will be overwritten to {value}')
        
        self.__log_dict[key] = value