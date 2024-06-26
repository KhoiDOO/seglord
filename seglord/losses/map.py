from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, Module

def get_loss(args) -> Module:

    mode = 'multiclass' if args.cls != 1 else 'binary'

    # y_pred: torch.Tensor, y_true: torch.Tensor
    
    if args.loss == 'ce':
        return CrossEntropyLoss() if mode == 'multiclass' else BCEWithLogitsLoss()
    elif args.loss == 'dice':
        return DiceLoss(mode=mode)
    elif args.loss == 'logdice':
        return DiceLoss(mode=mode, log_loss=True)
    elif args.loss == 'jaccard':
        return JaccardLoss(mode=mode)
    elif args.loss == 'logjaccard':
        return JaccardLoss(mode=mode, log_loss=True)