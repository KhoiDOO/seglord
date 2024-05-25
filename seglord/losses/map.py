from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss


def get_loss(args):

    mode = 'multiclass' if args.cls != 1 else 'binary'
    
    if args.loss == 'ce':
        pass
    elif args.loss == 'dice':
        return DiceLoss(mode=mode)
    elif args.loss == 'logdice':
        return DiceLoss(mode=mode, log_loss=True)
    elif args.loss == 'jaccard':
        return JaccardLoss(mode=mode)
    elif args.loss == 'logjaccard':
        return JaccardLoss(mode=mode, log_loss=True)