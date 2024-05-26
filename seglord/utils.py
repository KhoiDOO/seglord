from typing import List, Dict
from torch import Tensor
from torch.nn import Module

from datetime import datetime

# import json
# import hashlib
# import lzma
# import torch
# import pickle
# import gc
import argparse
import os

def folder_setup(args: argparse):
    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    args.runid = now

    run_dir = os.getcwd() + "/runs"
    save_dir = run_dir + f"/{now}"
    os.makedirs(save_dir, exist_ok=True)

    args.svdir = save_dir

    return save_dir, args

def param_cnt(model: Module):
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    return pytorch_total_params