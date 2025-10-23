import torch.nn as nn
from torch.cuda.amp import autocast,GradScaler
from torch import load
from torch.optim import Adam, lr_scheduler
from torch.nn import init,Conv2d,Conv3d,ConvTranspose2d
from numpy import prod

# from networks.unet import UNet 
# from networks.resnet import ResUNet

class FLFMnet(nn.Module):
    def __init__(self,Recon_Net, in_views, output_shape,
                 dataset=None, use_bias=False,
                 unet_settings={'depth':5, 'wf':6, 'drop_out':1.0, 'batch_norm':True}):
        super(FLFMnet, self).__init__()
        self.output_shape = output_shape

        self.dataset = dataset
        out_depths = output_shape[2]
        
        # 3D reconstruction net
        self.deconv = nn.Sequential(
                        nn.Conv2d(in_views,out_depths, 3, stride=1, padding=1, 
                                  bias=use_bias),
                        nn.BatchNorm2d(out_depths),
                        nn.LeakyReLU(),
                        Recon_Net(out_depths, out_depths, depth=unet_settings['depth'], 
                               wf=unet_settings['wf'], drop_out=unet_settings['drop_out'], 
                               use_bias=use_bias))
    @autocast()
    def forward(self, input):
        # Fetch normalization stats for SLNet
        imputimg = input#['curr_img_stack']
        # inputvol = input['local_volumes']
        intermediate_result = self.dataset.extract_views(imputimg[:,0,...].unsqueeze(1),
                                                         self.dataset.lenslet_coords, 
                                                         self.dataset.subimage_shape,
                                                         self.dataset.data_scale)[:,0,...]
        # Run 3D reconstruction network
        out = self.deconv(intermediate_result)            
        return out, intermediate_result

class setupNetwork(nn.Module):
    def __init__(self, Recon_Net, dataset, args, stats, device):
        super(setupNetwork, self).__init__()
        args,checkpoint_FLFMnet = self.configUNet(args, device)
        net = FLFMnet(Recon_Net,dataset.n_lenslets, args.output_shape, 
                      dataset=dataset, unet_settings=args.unet_settings).to(device)
        # net = compile(net0)
        net.apply(self.init_weights)
        print("# Weights initialization function created!")
        
        trainable_params = [{'params': net.deconv.parameters()}]
        params = sum([prod(p.size()) for p in net.parameters()])

        optimizer = Adam(trainable_params, lr=args.learning_rate)
        lr = args.learning_rate
        lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        scaler = GradScaler()
        print("# Optimizer created", optimizer)
        net.stats = stats

        if args.train_ckpt_from!=None:
            net.load_state_dict(checkpoint_FLFMnet['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint_FLFMnet['optimizer_state_dict'])
            self.start_epoch = checkpoint_FLFMnet['epoch']-1
            print("# Loaded checkpoint from ", args.train_ckpt_from)
        else:
            self.start_epoch = 0
            print('# No previous checkpoint found, starting from scratch')

        self.values2return = (net,checkpoint_FLFMnet,optimizer,lr,scaler,lr_sched,params)

    def configUNet(self,args, device):
        # Load previous checkpoints
        if args.train_ckpt_from!=None:
            checkpoint_FLFMnet = load(args.train_ckpt_from, map_location=device)
            args_deconv = checkpoint_FLFMnet['args']
            args.unet_depth = args_deconv.unet_depth
            args.unet_wf = args_deconv.unet_wf
        else:
            checkpoint_FLFMnet = None

        unet_settings = {'depth':args.unet_depth, 'wf':args.unet_wf, 'drop_out':args.unet_dropout}
        args.unet_settings = unet_settings
        print("# Unet settings: ", unet_settings)

        return args,checkpoint_FLFMnet
    
    def init_weights(self,m):
        if type(m) == Conv2d or type(m) == Conv3d or type(m) == ConvTranspose2d:
            init.xavier_uniform(m.weight)
