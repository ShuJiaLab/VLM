import torch
import os
from rich.console import Console
console = Console(color_system="truecolor",style=None)
os.system('cls' if os.name == 'nt' else 'clear')

import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import os
from train.trainNetwork import normalize_type

from datapreproc.FLFMDataset import FLFMDataset
from imageio import volwrite

from utilities.util_setupSystem import *
from utilities.util_setupPSFOTF import setupPSFOTF
from networks.FLFMnet import FLFMnet


YMLFILENAME = './paraymls/STORM_microtubule.yml' # set up parameter file
TrainingPrefix = '3dunet-d7prediction' # prefix for training ID, make notes to the training session
recalcPSFcenters = True # recalculate lenslet centers
original_image_shape = [1024,1024]
original_subimage_shape = [1024,1024]
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

# Load previous checkpoints
if len(args.predict_ckpt_from)>0:
    checkpoint_FLFMNet = torch.load(args.predict_ckpt_from, map_location=device)
    args_deconv = checkpoint_FLFMNet['args']

OTF, psf_shape = setupPSFOTF(args, device, args.img_depth,recalcPSFcenters=recalcPSFcenters) # set up PSF and OTF



imagerangemax = 1000 #args.imagerangemax
if (args.predict_img_end-args.predict_img_start+1)>imagerangemax:
    print(">>> Splitting reconstruction into multiple batches",imagerangemax,"images each")
    totalimgnum = args.predict_img_end-args.predict_img_start+1
    imgs2use = [range(imagerangemax*ii+args.predict_img_start-1,\
                      min(imagerangemax*(ii+1),totalimgnum)+args.predict_img_start-1) \
                        for ii in range(0,int(float(totalimgnum)//imagerangemax+1))]
    print(">>> Reconstruction will be split into ",imgs2use,"batches")
else:
    imgs2use = [range(args.predict_img_start-1,args.predict_img_end)]
''' ===================================================================================================='''

for imgsubrange in imgs2use:
    print("# Reconstructing images: ", imgsubrange[0]+1,"-",imgsubrange[-1]+1,"/")
    # Create dataloaders
    dataset = FLFMDataset(args.predict_folder_in, args.test_folder_gt,args.lenslet_file,subimage_shape, img_shape,
                        args.data_rescale,images_to_use=imgsubrange, n_depths_to_fill=args.img_depth,load_vols=False)
    dataset_size = len(dataset)
    test_indices = list(range(dataset_size))
    train_sampler = SequentialSampler(dataset)
    test_loader = data.DataLoader(dataset,sampler=train_sampler, num_workers=0, shuffle=False)
    print("# Dataset created, size:", dataset_size)

    # Get normalization values 
    # max_images,max_images_sparse,max_volumes = dataset.get_max() 
    stats = checkpoint_FLFMNet['statistics'] #dataset.get_statistics()

    # Create net
    if networkoption == 'UNet':
        from networks.unet import UNet as Recon_Net
    elif networkoption == 'ResUNet':
        from networks.resnet import ResUNet as Recon_Net
    elif networkoption == 'UNet3d':
        from networks.unet3d import UNet3d as Recon_Net
        
    net = FLFMnet(Recon_Net,dataset.n_lenslets, args_deconv.output_shape, dataset=dataset,
                  unet_settings=args_deconv.unet_settings).to(device)

    # timers
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start_epoch = 0
    if len(args.predict_ckpt_from)>0:
        net.load_state_dict(checkpoint_FLFMNet['model_state_dict'], strict=False)


    # if args.writeVolsToStack:
    #     if not os.path.exists(save_folder):
    #         os.makedirs(save_folder)

    # Update noramlization stats for SLNet inside network
    net.stats = stats
    net = net.eval()

    plt.ion()
    fig, ax = plt.subplots()
    imshowHDL = ax.imshow(dataset.stacked_views[0,:,:])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(5)

    with torch.no_grad():
        for ix,(curr_img_stack, local_volumes) in enumerate(test_loader):
            
            curr_img_stack = curr_img_stack.half().to(device)
            local_volumes = local_volumes.half().to(device)

            curr_img_stack = curr_img_stack - args.predict_baseline_in
            curr_img_stack = F.relu(curr_img_stack).detach()
            curr_img_stack, _ = normalize_type(curr_img_stack, local_volumes, 
                                            stats['norm_type_img'], stats['mean_imgs'], stats['std_images'], 
                                            stats['mean_vols'], stats['std_vols'], stats['max_images'], stats['max_vols'])     

            with autocast():
            # if True:
                start.record()
        
                # Run batch of predicted images in discriminator
                networkinput = curr_img_stack #{'curr_img_stack':curr_img_stack,'local_volumes':local_volumes}
                prediction,sparse_prediction = net(networkinput)

                if not all([prediction.shape[i] == subimage_shape[i-2] for i in range(2,4)]):
                    diffY = (subimage_shape[0] - prediction.size()[2])
                    diffX = (subimage_shape[1] - prediction.size()[3])

                    prediction = F.pad(prediction, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))


                pred_proj = prediction[0,...].max(dim=0).values
                pred_proj = pred_proj/pred_proj.max()*60000
                print(pred_proj.shape, pred_proj.min(), pred_proj.max())
                imshowHDL.set_data(pred_proj.cpu().numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()
                # time.sleep(1)

                # Record training time
                end.record()
                torch.cuda.synchronize()
                end_time = start.elapsed_time(end)
                print(ix, "--" ,prediction[0,...].shape, " time:", round(end_time,2), "ms | Freq:", round(1000/end_time,2), "Hz")
                
                if args.writeVolsToStack>0:
                    stack_to_save = prediction[0,...].cpu().numpy().squeeze()
                    # stack_to_save = (stack_to_save - stack_to_save.min())/(stack_to_save.max()-stack_to_save.min())*60000
                    stack_to_save = stack_to_save.astype(np.float16)
                    print(ix, "--" ,stack_to_save.shape, " time:", round(end_time,2), "ms | Freq:", round(1000/end_time,2), "Hz")
                    volwrite(args.output_folder + '/FLFM_stack_'+ "%05d" % (ix+1+imgsubrange[0]) + '.tif', stack_to_save)


