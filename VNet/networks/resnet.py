from torch import nn,cat
import torch.nn.functional as F
from torch.cuda.amp import autocast #,GradScaler
from networks.blocks import ResidualBlock

class ResUNet(nn.Module):
    def __init__(self,in_channels=1,n_classes=2,depth=5,wf=6,padding=True,batch_norm=True,up_mode='upconv',
        drop_out=0,use_bias=False,):
        """
        Implementation of
        Res-U-Net: RoadExtractionbyDeepResidualU-Net
        (Zhengming Zhang et al., 2017)
        https://arxiv.org/abs/1711.10684
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(ResUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.drop_out = drop_out
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(ResidualBlock(3, prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(ResUNetUpBlock(prev_channels, 2 ** (wf + i), up_mode))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(nn.Conv2d(prev_channels, n_classes, kernel_size=1, bias=use_bias),
                    nn.ReLU()
        )

    @autocast()
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
                x = F.dropout2d(x, self.drop_out)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            x = F.dropout2d(x, self.drop_out)

        return self.last(x)


class ResUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode):
        super(ResUNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=False)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = ResidualBlock(3,in_size, out_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    