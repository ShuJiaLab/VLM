from os import system
from os.path import isfile
from sys import exit
import time
from rich.console import Console
import torch
from torch.nn.functional import relu as Frelu
from torch.nn.functional import pad as Fpad
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from utilities.util_str2url import getTBurl
from utilities.util_imVisualizeSave import *
from utilities.util_notice import send_notice
from utilities.util_camnoise import add_camera_noise
from utilities.util_reprojection import *
from pytorch_msssim import SSIM

def net_get_params(net):
    if hasattr(net, 'module'):
        return net.module
    else:
        return net

# Apply different normalizations to volumes and images
def normalize_type(LF_views, vols, id=0, mean_imgs=0, std_imgs=1, mean_vols=0, std_vols=1, max_imgs=1, max_vols=1, inverse=False):
    if inverse:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) * (2*std_imgs), vols * std_vols + mean_vols
        if id==1: # Standarization of images and volume normalization
            return LF_views * std_imgs + mean_imgs, vols * std_vols
        if id==2: # normalization of both
            return LF_views * max_imgs, vols * max_vols
        if id==3: # normalization of both
            return LF_views * std_imgs, vols * std_vols
    else:
        if id==-1: # No normalization
            return LF_views, vols
        if id==0: # baseline normlization
            return (LF_views) / (2*std_imgs), (vols - mean_vols) / std_vols
        if id==1: # Standarization of images and volume normalization
            return (LF_views - mean_imgs) / std_imgs, vols / std_vols
        if id==2: # normalization of both
            return LF_views / max_imgs, vols / max_vols
        if id==3: # normalization of both
            return LF_views / std_imgs, vols / std_vols
          
def write2TB(args,writer,epoch,local_volumes,prediction,
             curr_train_stage,reproj,curr_views,
             mean_repro_ssim,mean_repro,mean_volume_loss,mean_psnr,mean_time,
             net,optimizer,scaler,lr,end_time,stats,save_folder):
    console = Console(color_system="truecolor",style=None)
    if epoch%5==0:
        if local_volumes.shape == prediction.shape:
            writer.add_image('max_GT_'+curr_train_stage, 
                            make_grid(volume_2_projections(local_volumes.permute(0,2,3,1).unsqueeze(1))[0,...], 
                            normalize=True, scale_each=True), epoch)
            
        writer.add_image('max_prediction_'+curr_train_stage, 
                        make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,...], 
                        normalize=True, scale_each=True), epoch)
        
        if curr_train_stage=='test' and len(args.gpu_repro)>0:
            repro_grid = make_grid(reproj[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), 
                        normalize=True, scale_each=False)
            writer.add_image('reproj_'+curr_train_stage, repro_grid, epoch)
            writer.add_image('reproj_GT_'+curr_train_stage, repro_grid, epoch)
            repro_grid = make_grid((curr_views-reproj)[0,2,...].abs().float().unsqueeze(0).cpu().data.detach(), 
                        normalize=True, scale_each=False)
            writer.add_image('reproj_error_'+curr_train_stage, repro_grid, epoch)
            writer.add_scalar('reproj/ssim/'+curr_train_stage, mean_repro_ssim, epoch)
            writer.add_scalar('reproj/Loss/'+curr_train_stage, mean_repro, epoch)

        writer.add_scalar('Loss/'+curr_train_stage,mean_volume_loss, epoch)
        # writer.add_scalar('psnr/',{curr_train_stage: mean_psnr}, epoch)
        # writer.add_scalar('times/',{curr_train_stage: mean_time/1000}, epoch)
        # writer.add_scalar('lr/',{curr_train_stage: lr}, epoch)
        writer.add_scalar('psnr/'+curr_train_stage, mean_psnr, epoch)
        writer.add_scalar('times/'+curr_train_stage, mean_time/1000, epoch)
        writer.add_scalar('lr/'+curr_train_stage, lr, epoch)
    
    if curr_train_stage=='train':
        console.print('[white]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(mean_time/1000,2)) + " s.")
    elif curr_train_stage=='val':
        console.print('[yellow]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(mean_time/1000,2)) + " s.")
        # send_notice('Epoch Number '+str(epoch) + ' at ' + curr_train_stage + " stage, loss is" + str(round(mean_volume_loss,5)))
    elif curr_train_stage=='test':
        console.print('[green]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(mean_time/1000,2)) + " s.")

    if isfile('./'+'exit_file.txt'):
        torch.cuda.empty_cache()
        exit(0)

    if epoch%args.val_every==0 and epoch!=0:
        torch.save({
        'epoch': epoch,
        'args' : args,
        'args_SLNet' : 'No argsSLNet',
        'statistics' : stats,
        'model_state_dict': net_get_params(net).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'loss': mean_volume_loss},
        save_folder + '/model_current')

    if epoch%(args.val_every)==0 and epoch!=0:
        torch.save({
        'epoch': epoch,
        'args' : args,
        'args_SLNet' : 'No argsSLNet',
        'statistics' : stats,
        'model_state_dict': net_get_params(net).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'loss': mean_volume_loss},
        save_folder + '/model_'+str(epoch))

def inputbkg_est(curr_img_stack):
    bkg_btmright = curr_img_stack[...,-5:,-5:].flatten(start_dim = -2)
    bkg_btmleft = curr_img_stack[...,-5:,:5].flatten(start_dim = -2)
    bkg_topright = curr_img_stack[...,:5,-5:].flatten(start_dim = -2)
    bkg_topleft = curr_img_stack[...,:5,:5].flatten(start_dim = -2)
    bkg = torch.cat((bkg_btmright,bkg_btmleft,bkg_topright,bkg_topleft),dim=-1).topk(25,dim=-1).values.median(-1).values
    return bkg

def trainNetwork(args,start_epoch,dataset,data_loaders,stats,
                 subimage_shape,OTF,psf_shape,
                 net,optimizer,scaler,lr_sched,
                 loss,ssimloss_module,
                 writer,save_folder,
                 start,end,device,device_repro):
    
    ssim_module = SSIM(data_range=1, size_average=True, channel=dataset.n_lenslets).to(device_repro)

    for epoch in range(start_epoch, args.max_epochs):
        for curr_train_stage in ['train','val','test']:
            # if curr_train_stage=='train' and epoch== 2:
                # system('start '+getTBurl("localhost")+'\n')
            # Grab current data_loader
            curr_loader = data_loaders[curr_train_stage]
            curr_loader_len = curr_loader.sampler.num_samples \
                            if curr_train_stage=='test' else len(curr_loader.batch_sampler.sampler.indices)

            if curr_train_stage=='train':
                net.train()
                torch.set_grad_enabled(True)
            if curr_train_stage=='val' or curr_train_stage=='test':
                if epoch%args.val_every!=0:continue
                net.eval()
                torch.set_grad_enabled(False)

            # Store loss
            mean_volume_loss = 0 
            mean_psnr = 0
            mean_time = 0
            mean_repro = 0
            mean_repro_ssim = 0

            # initialize training variables
            lr = args.learning_rate
            reproj,curr_views = None,None

            # Training
            for ix,(curr_img_stack, local_volumes) in enumerate(curr_loader):
                # Normalize volumes if ill posed
                signal_power_rand = torch.rand(1)
                # if local_volumes.float().max()>=10000:#(args.train_baseline_gt+1):
                local_volumes = Frelu(local_volumes.float()).detach()
                local_volumes = local_volumes / local_volumes.max() * 10000.0
                local_volumes = local_volumes.half()
                local_volumes = local_volumes.to(device)

                curr_img_stack_bkg = args.train_baseline_in if curr_train_stage=='test' \
                    else inputbkg_est(curr_img_stack).unsqueeze(-1).unsqueeze(-1)
                curr_img_stack = curr_img_stack - curr_img_stack_bkg
                curr_img_stack = Frelu(curr_img_stack).detach()

                if args.add_noise==True:
                    curr_max = curr_img_stack.max()
                    curr_min = curr_img_stack.min()
                    if curr_train_stage!='test':
                        signal_power = (args.signal_min + (args.signal_max-args.signal_min) * signal_power_rand).item()
                        curr_img_stack = signal_power/(curr_max-curr_min) * (curr_img_stack-curr_min)
                        curr_img_stack = add_camera_noise(curr_img_stack)
                    else:
                        # signal_power = (args.signal_min + (args.signal_max-args.signal_min) * signal_power_rand).item()
                        # curr_img_stack = signal_power/(curr_max-curr_min) * (curr_img_stack-curr_min)
                        curr_img_stack = curr_img_stack
                        # curr_img_stack = add_camera_noise(curr_img_stack)
                    curr_img_stack = curr_img_stack.float().to(device)

                # if conversion to half precission messed up the volumes, continue
                if torch.isinf(local_volumes.max()):
                    curr_loader_len -= local_volumes.shape[0]
                    continue

                # Images are already normalized from mainCreateDataset.py
                curr_img_stack, local_volumes = normalize_type(curr_img_stack, local_volumes,stats['norm_type'], 
                                                                stats['mean_imgs'], stats['std_images'], 
                                                                stats['mean_vols'], stats['std_vols'], 
                                                                stats['max_images'], stats['max_vols'])

                # ================== start recording time ==================
                start.record()

                if curr_train_stage=='train':
                    net.zero_grad()
                    optimizer.zero_grad()
                # 
                with autocast():
                    # Run batch of predicted images in discriminator
                    # networkinput = {'curr_img_stack':curr_img_stack,'local_volumes':local_volumes}
                    networkinput = curr_img_stack
                    prediction,sparse_prediction = net(networkinput)

                    if not all([prediction.shape[i] == subimage_shape[i-2] for i in range(2,4)]):
                        diffY = (subimage_shape[0] - prediction.size()[2])
                        diffX = (subimage_shape[1] - prediction.size()[3])

                        prediction = Fpad(prediction, (diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2))

                    # print(prediction.shape, local_volumes.shape)
                    volume_loss = loss(local_volumes, prediction,ssimloss_module)
                    

                    if curr_train_stage=='test' and len(args.gpu_repro)>0:
                        with torch.no_grad():
                            reproj_loss, reproj,curr_views,_ = reprojection_loss(sparse_prediction, prediction.float(), 
                                                            OTF, psf_shape, dataset, n_split=1, device=device_repro)
                        mean_repro += reproj_loss.item()
                        mean_repro_ssim += ssim_module((sparse_prediction/sparse_prediction.max()).to(device_repro).float(), 
                                                    (reproj/reproj.max()).float().to(device_repro)).cpu().item()
                    
                    if curr_train_stage=='train' and epoch == 0:reproj,curr_views = None,None
                    
                mean_volume_loss += volume_loss.mean().detach().item()

                if curr_train_stage=='train':
                    scaler.scale(volume_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                        

                # Record training time
                # end_each_time = round(time.time() - start_time,2)
                end.record()
                torch.cuda.synchronize()
                end_time = start.elapsed_time(end)
                mean_time += end_time

                # detach tensors
                local_volumes = local_volumes.detach().cpu().float()
                prediction = prediction.detach().cpu().float()
                curr_img_stack = curr_img_stack.detach()


                if torch.isinf(torch.tensor(mean_volume_loss)): print('inf')

                local_volumes -= local_volumes.min()
                prediction -= prediction.min()

                prediction /= stats['max_vols']
                local_volumes /= stats['max_vols']

                curr_img_stack -= curr_img_stack.min()
                curr_img_stack /= curr_img_stack.max()
            
            mean_volume_loss /= curr_loader_len
            mean_psnr = 20 * torch.log10(stats['max_vols'] / torch.sqrt(torch.tensor(mean_volume_loss))) #/= curr_loader_len
            # mean_time /= curr_loader_len
            mean_repro /= curr_loader_len
            mean_repro_ssim /= curr_loader_len

            # Update learning rate
            # if curr_train_stage=='train' and epoch == 0:
            #     lr = args.learning_rate
            if (epoch-start_epoch)>100:
                lr_sched.step()
                lr = optimizer.param_groups[0]['lr']
                args.learning_rate = lr
            else:
                lr = args.learning_rate
            # if curr_train_stage=='val' and epoch>100:
            #     lr_sched.step(mean_volume_loss)
            #     lr = optimizer.param_groups[0]['lr']
            #     args.learning_rate = lr
            # if curr_train_stage!='val' and epoch>101:
            #     lr = args.learning_rate            

            write2TB(args,writer,epoch,local_volumes,prediction,
                    curr_train_stage,reproj,curr_views,
                    mean_repro_ssim,mean_repro,mean_volume_loss,mean_psnr,mean_time,
                    net,optimizer,scaler,lr,end_time,stats,save_folder)