# seglord: Segmentation Lord

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