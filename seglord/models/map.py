from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, PAN, DeepLabV3, DeepLabV3Plus

def unet(args) -> Unet:
    return Unet(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_use_batchnorm=args.bn,
        decoder_channels=args.dchannels,
        decoder_attention_type=args.att,
        in_channels=args.inc,
        classes=args.cls
    )

def unetpp(args) -> UnetPlusPlus:
    return UnetPlusPlus(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_use_batchnorm=args.bn,
        decoder_channels=args.dchannels,
        decoder_attention_type=args.att,
        in_channels=args.inc,
        classes=args.cls
    )

def manet(args) -> MAnet:
    return MAnet(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_use_batchnorm=args.bn,
        decoder_channels=args.dchannels,
        decoder_pab_channels=args.pab,
        in_channels=args.inc,
        classes=args.cls
    )

def lnet(args) -> Linknet:
    return Linknet(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_use_batchnorm=args.bn,
        in_channels=args.inc,
        classes=args.cls
    )

def fpn(args) -> FPN:
    return FPN(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_pyramid_channels=args.prm,
        decoder_segmentation_channels=args.segch,
        decoder_merge_policy=args.decmerge,
        in_channels=args.inc,
        classes=args.cls,
        upsampling=args.ups
    )

def psp(args):
    return PSPNet(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        psp_out_channels=args.pspoch,
        psp_use_batchnorm=args.bn,
        in_channels=args.inc,
        classes=args.cls,
        upsampling=args.ups
    )

def pan(args):
    return PAN(
        encoder_name=args.ename,
        encoder_weights=args.eweight,
        encoder_output_stride=args.estride,
        decoder_channels=args.dchannels,
        in_channels=args.inc,
        classes=args.cls,
        upsampling=args.ups
    )

def dl3(args) -> DeepLabV3:
    return DeepLabV3(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        decoder_channels=args.dchannels,
        in_channels=args.inc,
        classes=args.cls,
        upsampling=args.ups
    )

def dl3p(args) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        encoder_name=args.ename,
        encoder_depth=args.edepth,
        encoder_weights=args.eweight,
        encoder_output_stride=args.estride,
        decoder_channels=args.args.dchannels,
        decoder_atrous_rates=args.drates,
        in_channels=args.inc,
        classes=args.cls,
        upsampling=args.ups
    )

mapping = {
    'unet' : unet,
    'unetpp' : unetpp,
    'manet' : manet,
    'lnet' : lnet,
    'fpn' : fpn,
    'psp' : psp,
    'pan' : pan,
    'dl3' : dl3,
    'dl3p' : dl3p
}

def get_model(args) -> Unet | UnetPlusPlus | MAnet | Linknet | FPN | PSPNet | PAN | DeepLabV3 | DeepLabV3Plus:
    return mapping[args.model](args)