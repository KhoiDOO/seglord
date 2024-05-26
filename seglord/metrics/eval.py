from torch import Tensor
from torch.nn import Module
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel

from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from torchmetrics.classification import \
    Accuracy, ConfusionMatrix, BinaryAccuracy, \
    F1Score, BinaryF1Score, \
    AUROC, BinaryAUROC, \
    ROC, BinaryROC, \
    Recall, BinaryRecall, \
    Specificity, BinarySpecificity

from einops import rearrange

from typing import List

import argparse
import torch
import wandb


class Evaluator:
    def __init__(self, args: argparse) -> None:
        self.__log_dict = {}

        self.cls = args.cls
        self.trldcnt = args.trldcnt
        self.vlldcnt = args.vlldcnt
        self.vb = args.verbose
        self.wb = args.wandb
        self.args = args
        self.svdir = args.svdir
        self.me = args.me
        self.old_metric_value = 1e26 if self.me in ['loss'] else 0
        
        self.cpfunc = self.compare_func()
        self.best_path = self.svdir + "/best.pt"
        self.last_path = self.svdir + "/last.pt"

        if self.wb:
            self.run = wandb.init(
                project=args.wandb_prj,
                entity=args.wandb_entity,
                config=args,
                name=args.runid,
                force=True
            )

    def __call__(self, criterion: Module, y_pred: Tensor, y_true: Tensor, mode: str) -> Tensor:
        """_summary_

        Args:
            criterion (Module): loss function module
            y_pred (Tensor): prediction with shape (B, C, H ,W)
            y_true (Tensor): ground-truth with shape (B, C, H, W)
            mode (str): 'train' or 'valid

        Returns:
            Tensor: loss for backward
        """

        bcnt = self.trldcnt if mode == 'train' else self.vlldcnt
        task = 'multiclass' if self.cls != 1 else 'binary'

        _y_pred = rearrange(y_pred, 'b c h w -> b h w c').flatten(0, 2)
        _y_true = rearrange(y_true, 'b c h w -> b h w c').flatten(0, 2)

        _y_pred = torch.argmax(_y_pred, dim=1) if mode == 'train' else torch.sigmoid(_y_pred)
        _y_true = torch.argmax(_y_true, dim=1) if mode == 'train' else _y_true
        
        # loss
        loss = criterion(y_pred, y_true)
        self.update(value=loss, mode=mode, metric_name='loss', bcnt=bcnt)

        # miou
        miou = MeanIoU(num_classes=self.cls)(y_pred, y_true)
        self.update(value=miou, mode=mode, metric_name='miou', bcnt=bcnt)
        
        # dice
        dice = GeneralizedDiceScore(num_classes=self.cls)(y_pred, y_true)
        self.update(value=dice, mode=mode, metric_name='dice', bcnt=bcnt)

        # acc
        acc = Accuracy(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=acc, mode=mode, metric_name='acc', bcnt=bcnt)

        # f1
        f1 = F1Score(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=f1, mode=mode, metric_name='f1', bcnt=bcnt)

        # auc
        auc = AUROC(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=auc, mode=mode, metric_name='auc', bcnt=bcnt)

        # roc
        roc = ROC(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=roc, mode=mode, metric_name='roc', bcnt=bcnt)

        # sen
        sens = Recall(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=sens, mode=mode, metric_name='sen', bcnt=bcnt)

        # spe
        spec = Specificity(task=task, num_classes=self.cls)(_y_pred, _y_true)
        self.update(value=spec, mode=mode, metric_name='spe', bcnt=bcnt)

        if task == 'multiclass':

            # clswise miou
            mious = MeanIoU(num_classes=self.cls, per_class=True)(y_pred, y_true)
            self.clswise_update(values=mious, mode=mode, metric_name='miou', bcnt=bcnt)

            # clswise dice
            dices = GeneralizedDiceScore(num_classes=self.cls, per_class=True)(y_pred, y_true)
            self.clswise_update(values=dices, mode=mode, metric_name='dice', bcnt=bcnt)

            # clswise acc
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinarySpecificity(), mode=mode, metric_name='spe', bcnt=bcnt)

            # clswise acc
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinaryAccuracy(), mode=mode, metric_name='acc', bcnt=bcnt)

            # clswise f1 
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinaryF1Score(), mode=mode, metric_name='f1', bcnt=bcnt)

            # clswise sens 
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinaryRecall(), mode=mode, metric_name='sen', bcnt=bcnt)

            # clswise auc 
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinaryAUROC(), mode=mode, metric_name='auc', bcnt=bcnt)

            # clswise roc 
            self.clswise_update_metric(y_pred=_y_pred, y_true=_y_true, fn=BinaryROC(), mode=mode, metric_name='roc', bcnt=bcnt)

        return loss

    @property
    def log(self):
        return self.__log_dict

    @property
    def txt(self):
        if self.vb == 0:
            return 'non verbose'
        elif self.vb == 1:
            return " - ".join([f'{key}: {value}:.2f' for key, value in self.__log_dict.items() if '-' not in key])
        elif self.vb == 2:
            return " - ".join([f'{key}: {value}:.2f' for key, value in self.__log_dict.items()])
    
    def step(self, evaluator: Accelerator, model: Module | DistributedDataParallel):
        
        if self.wb:
            self.run.log(self.__log_dict)
        
        unwrap_model: Module = evaluator.unwrap_model(model)
        
        
        if self.cpfunc(self.__log_dict[f'valid/{self.me}'], self.old_metric_value):
            pass
        
        self.__log_dict = {}

    def clswise_update(self, values: Tensor | List, mode: str, metric_name: str, bcnt: int):
        
        values = values.tolist() if isinstance(Tensor) else values
        
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
            bcnt (int): number of batchs
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