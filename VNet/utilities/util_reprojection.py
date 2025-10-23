from torch.nn.functional import mse_loss as F_mse_loss
from torch import zeros_like as torch_zeros_like
from utilities.util_setupPSFOTF import fft_conv_split


def reprojection_loss_camera(gt_imgs, prediction, PSF, camera, dataset, device="cpu"):
    out_type = gt_imgs.type()
    camera = camera.to(device)
    reprojection = camera(prediction.to(device), PSF.to(device))
    reprojection_views = dataset.extract_views(reprojection,dataset.lenslet_coords, dataset.subimage_shape,dataset.data_scale)[0,0,...]
    loss = F_mse_loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

def reprojection_loss(gt_imgs, prediction, OTF, psf_shape, dataset, n_split=1, device="cpu", loss=F_mse_loss):
    out_type = gt_imgs.type()
    batch_size = prediction.shape[0]
    reprojection = fft_conv_split(prediction[0,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)

    reprojection_views = torch_zeros_like(gt_imgs)
    reprojection_views[0,...] = dataset.extract_views(reprojection,dataset.lenslet_coords, dataset.subimage_shape,dataset.data_scale)[0,0,...]

    # full_reprojection = reprojection.detach()
    # reprojection_views = reprojection_views.unsqueeze(0).repeat(batch_size,1,1,1)
    for nSample in range(1,batch_size):
        reprojection = fft_conv_split(prediction[nSample,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)
        reprojection_views[nSample,...] = dataset.extract_views(reprojection,dataset.lenslet_coords, dataset.subimage_shape,dataset.data_scale)[0,0,...]
        # full_reprojection += reprojection.detach()

    # gt_imgs /= gt_imgs.float().max()
    # reprojection_views /= reprojection_views.float().max()
    # loss = F.mse_loss(gt_imgs[gt_imgs!=0].to(device), reprojection_views[gt_imgs!=0])
    #loss = (1-gt_imgs[reprojection_views!=0]/reprojection_views[reprojection_views!=0]).abs().mean()
    loss = loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)
