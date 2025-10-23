import torch
from numpy import float32 as npfloat32
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tifffile import imsave
from torch.nn.functional import interpolate

# Prepare a volume to be shown in tensorboard as an image
def volume_2_tensorboard(vol, batch_index=0, z_scaling=2):
    vol = vol.detach()
    # expecting dims to be [batch, depth, xDim, yDim]
    xyProj = make_grid(vol[batch_index,...].float().unsqueeze(0).sum(1).cpu().data, normalize=True, scale_each=True)
    
    # interpolate z in case that there are not many depths
    vol = torch.nn.functional.interpolate(vol.permute(0,2,3,1).unsqueeze(1), (vol.shape[2], vol.shape[3], vol.shape[1]*z_scaling))
    yzProj = make_grid(vol[batch_index,...].float().unsqueeze(0).sum(3).cpu().data, normalize=True, scale_each=True)
    xzProj = make_grid(vol[batch_index,...].float().unsqueeze(0).sum(2).cpu().data, normalize=True, scale_each=True)

    return xzProj, yzProj, xyProj


def volume_2_projections(vol_in, proj_type=torch.max, scaling_factors=[1,1,2], depths_in_ch=False, ths=[0.0,1.0], normalize=False, border_thickness=10, add_scale_bars=True, scale_bar_vox_sizes=[40,20]):
    vol = vol_in.detach().clone()
    # Normalize sets limits from 0 to 1
    if normalize:
        vol -= vol.float().min()
        vol /= vol.float().max()
    if depths_in_ch:
        vol = vol.permute(0,2,3,1).unsqueeze(1)
    if ths[0]!=0.0 or ths[1]!=1.0:
        vol_min,vol_max = vol.min(),vol.max()
        vol[(vol-vol_min)<(vol_max-vol_min)*ths[0]] = 0
        vol[(vol-vol_min)>(vol_max-vol_min)*ths[1]] = vol_min + (vol_max-vol_min)*ths[1]

    vol_size = list(vol.shape)
    vol_size[2:] = [vol.shape[i+2] * scaling_factors[i] for i in range(len(scaling_factors))]

    if proj_type is torch.max or proj_type is torch.min:
        x_projection, _ = proj_type(vol.float().cpu(), dim=2)
        y_projection, _ = proj_type(vol.float().cpu(), dim=3)
        z_projection, _ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = z_projection.min() * torch.ones(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4] + border_thickness, vol_size[3] + vol_size[4] + border_thickness
    )

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] + border_thickness :, : vol_size[3]] = interpolate(x_projection.permute(0, 1, 3, 2), size=[vol_size[-1],vol_size[-3]])
    out_img[:, :, : vol_size[2], vol_size[3] + border_thickness :] = interpolate(y_projection, size=[vol_size[2],vol_size[4]])

    line_color = out_img.max()
    # Draw white lines
    out_img[:, :, vol_size[2]: vol_size[2]+ border_thickness, ...] = line_color
    out_img[:, :, :, vol_size[3]:vol_size[3]+border_thickness, ...] = line_color

    if add_scale_bars:
        start = 0.02
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, int(0.9* vol_size[3]):int(0.9* vol_size[3])+scale_bar_vox_sizes[0]] = line_color
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2]] = line_color
        out_img[:, :, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2], int(start* vol_size[2]):int(start* vol_size[2])+4] = line_color

    return out_img

def imshow2D(img, blocking=False):
    plt.figure(figsize=(10,10))
    plt.imshow(img[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()

def imshow3D(vol, blocking=False):
    plt.figure(figsize=(10,10))
    plt.imshow(volume_2_projections(vol.permute(0,2,3,1).unsqueeze(1), normalize=True)[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
        
def imshowComplex(vol, blocking=False):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(volume_2_projections(torch.real(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    plt.subplot(1,2,2)
    plt.imshow(volume_2_projections(torch.imag(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()

def save_image(tensor, path='output.png'):
    if 'tif' in path:
        imsave(path, tensor[0,...].cpu().numpy().astype(npfloat32))
        return
    if tensor.shape[1] == 1:
        imshow2D(tensor)
    else:
        imshow3D(tensor)
    plt.savefig(path)

