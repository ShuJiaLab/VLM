import os
import torch
from rich.console import Console
console = Console(color_system="truecolor",style=None)
os.system('cls' if os.name == 'nt' else 'clear')
from utilities.util_setupSystem import *
from utilities.util_setupPSFOTF import setupPSFOTF
from datapreproc.FLFMDataset import FLFMDataset
from networks.FLFMnet import setupNetwork
from utilities.util_setupWriter import setupWriter
from utilities.util_setupLosses import setupLosses
from utilities.util_notice import send_notice
from train.trainNetwork import trainNetwork

''' ===================================================================================================='''
''' Use this part to change training settings.
    DefaultParams.yml is the default parameter file.'''
YMLFILENAME = './paraymls/STORM_microtubule.yml' # set up parameter file
TrainingPrefix = '3dunet-depth7' # prefix for training ID, make notes to the training session
original_image_shape = [1024,1024]
original_subimage_shape = [1024,1024]
recalcPSFcenters = True # recalculate lenslet centers
networkmap = ['UNet','ResUNet','UNet3d']
networkoption = networkmap[2]
''' ===================================================================================================='''

args = setupParams(YMLFILENAME,default=False) # set up parameters, default=True to use default parameters
device, device_repro,n_threads = setupDevices(args,useBenchmark=True) # set up devices
training_id, save_folder = setupTrainingID(args, YMLFILENAME, prefix=TrainingPrefix) # set up training ID

# set up training data shapes
img_shape = [round(original_image_shape[0]*args.data_rescale),round(original_image_shape[1]*args.data_rescale)]
subimage_shape = [round(original_subimage_shape[0]*args.data_rescale),round(original_subimage_shape[1]*args.data_rescale)]
args.output_shape = img_shape + [args.img_depth]

OTF, psf_shape = setupPSFOTF(args, device, args.img_depth,recalcPSFcenters=recalcPSFcenters) # set up PSF and OTF

console.rule('[bold red]# Loading dataset #')
dataset2train = FLFMDataset(args.train_folder_in, args.train_folder_gt,args.lenslet_file,subimage_shape, img_shape,args.data_rescale,
                            images_to_use=args.images_to_train, n_depths_to_fill=args.img_depth,load_vols=True)
dataset2test = FLFMDataset(args.test_folder_in, args.test_folder_gt, args.lenslet_file,subimage_shape, img_shape,args.data_rescale,
                           images_to_use=args.images_to_test, n_depths_to_fill=args.img_depth,load_vols=True)
dataset, dataset_test, data_loaders, stats = FLFMDataset.setupDataset(dataset2train,dataset2test,args,n_threads)
''' ===================================================================================================='''
console.rule('[bold red]# Creating network #')
if networkoption == 'UNet':
    from networks.unet import UNet as NETWORK_OBJ
elif networkoption == 'ResUNet':
    from networks.resnet import ResUNet as NETWORK_OBJ
elif networkoption == 'UNet3d':
    from networks.unet3d import UNet3d as NETWORK_OBJ
NETWORK_OBJ_set = setupNetwork(NETWORK_OBJ,dataset, args, stats, device)
net,checkpoint_FLFMnet,optimizer,lr,scaler,lr_sched,params = NETWORK_OBJ_set.values2return
print("# Network created, trainable parameters: ", params)
''' ===================================================================================================='''
console.rule('[bold red]# Creating summary writer to log stuff #')
# writer = setupWriter(args, save_folder,params, net, 
#                      {'curr_img_stack':torch.rand(1, 1, img_shape[0], img_shape[1]).to(device),
#                       'local_volumes':torch.rand(1, 16, img_shape[0], img_shape[1]).to(device)},YMLFILENAME)
writer = setupWriter(args, save_folder,params,net,torch.rand(1, 1, img_shape[0], img_shape[1]).to(device),YMLFILENAME)
''' ===================================================================================================='''
console.rule('[bold red]# Setting up losses #')
ssimloss_module,loss,loss_img = setupLosses(args, device)
''' ===================================================================================================='''
console.rule('[bold red]# Misc settings before training #')
if len(args.gpu_repro)>0:
    net.OTF_options =   {'OTF':OTF,'psf_shape':psf_shape,'dataset':dataset,'n_split':1,'loss_img':loss_img}
start_epoch = NETWORK_OBJ_set.start_epoch
print("# Training starts from epoch", start_epoch)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
os.system('start powershell -command "tensorboard --logdir=' + save_folder + ' --port=6006"')
''' ===================================================================================================='''
console.rule('[bold yellow]# Training begins! #')
send_notice('Training loop starts now!')
trainNetwork(args,start_epoch,dataset,data_loaders,stats,
             subimage_shape,OTF,psf_shape,
             net,optimizer,scaler,lr_sched,
             loss,ssimloss_module,
             writer,save_folder,
             start,end,device,device_repro)
