from typing import List, Dict
from torch import Tensor
from torch.nn import Module

from datetime import datetime

import json
import hashlib
import lzma
import torch
import pickle
import gc
import argparse
import os


def save_json(dct: Dict, path: str) -> None:
    with open(path, 'w') as outfile:
        json.dump(dct, outfile) 

def read_json(path: str) -> Dict:
    return json.load(open(path, 'r'))

def save_pickle_xz(dct: Dict, path:str) -> None:
    gc.disable()
    with lzma.open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    gc.enable()

def read_pickle_xz(path:str) -> Dict:
    gc.disable()
    with lzma.open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    gc.enable()
    return dct

def save_pickle(dct: Dict, path:str) -> None:
    with open(path, 'wb') as  file:
        pickle.dump(obj=dct, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path:str) -> None:
    with open(path, 'rb') as  file:
        dct = pickle.load(file=file)
    file.close()
    return dct

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