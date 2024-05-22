mkdir .cache

TMPDIR=./.cache pip install wheel tqdm
TMPDIR=./.cache pip3 install torch torchvision torchaudio
TMPDIR=./.cache pip install accelerate einops
TMPDIR=./.cache pip install albumentations
TMPDIR=./.cache pip install segmentation_models_pytorch
TMPDIR=./.cache pip install torchmetrics