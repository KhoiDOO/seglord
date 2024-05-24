from typing import List, Dict
from torch import Tensor

import json, hashlib, lzma, torch, pickle, gc


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