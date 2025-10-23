import os
os.system('cls' if os.name == 'nt' else 'clear')
import torch
from torchinfo import summary
from utilities.util_setupSystem import *
from networks.FLFMnet import FLFMnet
from datapreproc.FLFMDataset import FLFMDataset
networkmap = ['UNet','ResUNet','UNet3d']
networkoption = networkmap[2]

original_image_shape = [1024,1024]
original_subimage_shape = [1024,1024]
data_rescale = 0.5
img_depth = 64
device = "cuda:0"
n_lenslets = 3
img_shape = [round(original_image_shape[0]*data_rescale),round(original_image_shape[1]*data_rescale)]
subimage_shape = [round(original_subimage_shape[0]*data_rescale),round(original_subimage_shape[1]*data_rescale)]
output_shape = img_shape + [img_depth]
''' ===================================================================================================='''
YMLFILENAME = './paraymls/STORM_microtubule.yml' # set up parameter file
args = setupParams(YMLFILENAME,default=False)
dataset = FLFMDataset(args.predict_folder_in, args.test_folder_gt,args.lenslet_file,subimage_shape, img_shape,
                    args.data_rescale,images_to_use=range(0,2), n_depths_to_fill=img_depth,load_vols=False)

if networkoption == 'UNet':
    from networks.unet import UNet as Recon_Net
elif networkoption == 'ResUNet':
    from networks.resnet import ResUNet as Recon_Net
elif networkoption == 'UNet3d':
    from networks.unet3d import UNet3d as Recon_Net
unet_settings = {'depth':args.unet_depth, 'wf':args.unet_wf, 'drop_out':args.unet_dropout}
args.unet_settings = unet_settings
net = FLFMnet(Recon_Net,n_lenslets, output_shape,dataset=dataset, unet_settings=args.unet_settings)
# # inputs = {'curr_img_stack':torch.rand(1, 1, img_shape[0], img_shape[1]),'local_volumes':torch.rand(1, 16, img_shape[0], img_shape[1])}
# summary(net,input_data=torch.rand(1, 1, img_shape[0], img_shape[1]),verbose=2)
print(net)
