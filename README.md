# SegLord: Segmentation Lord

# Installation
## Environment Setup
```
python3 -m venv .env
source .env/bin/activate
python -m pip install -U pip
```

## Package Installation
```
bash install.sh
```
OR

```
mkdir .cache

TMPDIR=./.cache pip install wheel tqdm wandb
TMPDIR=./.cache pip3 install torch torchvision torchaudio
TMPDIR=./.cache pip install accelerate einops
TMPDIR=./.cache pip install albumentations
TMPDIR=./.cache pip install segmentation_models_pytorch
TMPDIR=./.cache pip install torchmetrics
TMPDIR=./.cache pip install segformer-pytorch
```

# Experiment Conduction
This repo training procedure is built with the support of ```Accelerator```, thus enabling various modes of training. Before training, direct the working folder to ```path/to/seglord/seglord```. There are two main ways for running with ```Accelerator```

```
accelerate launch {script_name.py} --arg1 --arg2 ...

or 

python -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

## Single GPU

```
CUDA_VISIBLE_DEVICES={GPU_ID} accelerate launch main.py --ds citynormal --model dl3p --loss dice --wandb 

or

accelerate launch --gpu_ids {GPU_ID} main.py --ds citynormal --model dl3p --loss dice --wandb 
```

## Multi GPUs

To use all available GPUs
```
accelerate launch --multi_gpu {GPU_ID} main.py --ds citynormal --model dl3p --loss dice --wandb 
```

Or to specify the number of GPUs in training
```
accelerate launch --num_processes=2 main.py --ds citynormal --model dl3p --loss dice --wandb
```

## CPU

To use CPU for training
```
accelerate launch --cpu main.py --ds citynormal --model dl3p --loss dice --wandb
```

## Precision Configuration
```
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 main.py --ds citynormal --model dl3p --loss dice --wandb
```

## Run from configs

There are templates for config file at ```path/to/seglord/seglord/scripts```.

### Single GPU

```
CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file ./scripts/single_gpu.yaml main.py --epochs 1 --debug --wandb
```

### Multi GPU

```
accelerate launch --config_file ./scripts/multi_gpu.yaml main.py --epochs 1 --debug --wandb
```