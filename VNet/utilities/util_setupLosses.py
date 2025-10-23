from utilities.util_lossfunctions import lossfunc
from pytorch_msssim import SSIM, MS_SSIM

def setupLosses(args, device):
    ssimloss_module = None
    if args.loss_type[:4] == 'ssim':
        ssimloss_module = SSIM(data_range=1, size_average=True, channel=args.img_depth).to(device)
    elif args.loss_type[:6] == 'ms_ssim':
        ssimloss_module = MS_SSIM(data_range=1, size_average=True, channel=args.img_depth).to(device)

    loss, loss_img = lossfunc(args.loss_type)
    print("# Loss function created")
    return ssimloss_module,loss,loss_img