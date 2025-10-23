import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def ssimlossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    return ssim_loss

def msssimlossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    return msssim_loss

def ssiml1lossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    return ssim_loss + l1_loss

def ssiml2lossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l2_loss = nn.MSELoss()(x,y)
    return ssim_loss + l2_loss

def ssimsmoothl1lossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return ssim_loss + smoothl1_loss

def msssiml1lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    return msssim_loss + l1_loss

def msssiml2lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l2_loss = nn.MSELoss()(x,y)
    return msssim_loss + l2_loss

def msssimsmoothl1lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return msssim_loss + smoothl1_loss

def ssiml1l2lossfunc(x,y,ssim_module):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    l2_loss = nn.MSELoss()(x,y)
    return ssim_loss + l1_loss + l2_loss

def msssiml1l2lossfunc(x,y,msssim_module):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    l2_loss = nn.MSELoss()(x,y)
    return msssim_loss + l1_loss + l2_loss

def l1lossfunc(x,y,ssim_module= None):
    l1_loss = nn.L1Loss()(x,y)
    return l1_loss

def l2lossfunc(x,y,ssim_module= None):
    l2_loss = nn.MSELoss()(x,y)
    return l2_loss

def smoothl1lossfunc(x,y,ssim_module= None):
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return smoothl1_loss
# ============================================================

def lossfunc(losstype,use_img_loss=1):   
    if losstype == 'l1':
        loss = l1lossfunc
    elif losstype == 'l2':
        loss = l2lossfunc
    elif losstype == 'smoothl1':
        loss = smoothl1lossfunc
    elif losstype == 'ssim':
        loss = ssimlossfunc
    elif losstype == 'msssim':
        loss = msssimlossfunc
    elif losstype == 'ssiml1':
        loss = ssiml1lossfunc
    elif losstype == 'ssiml2':
        loss = ssiml2lossfunc
    elif losstype == 'ssimsmoothl1':
        loss = ssimsmoothl1lossfunc
    elif losstype == 'msssiml1':
        loss = msssiml1lossfunc
    elif losstype == 'msssiml2':
        loss = msssiml2lossfunc
    elif losstype == 'msssimsmoothl1':
        loss = msssimsmoothl1lossfunc
    elif losstype == 'ssiml1l2':
        loss = ssiml1l2lossfunc
    elif losstype == 'msssiml1l2':
        loss = msssiml1l2lossfunc

    if use_img_loss>0:
        loss_img = loss
    else:
        loss_img = nn.MSELoss()
    return loss, loss_img

