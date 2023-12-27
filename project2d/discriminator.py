import torch.nn as nn

def get_downscaling_block(channels_in, channels_out, kernel, stride, padding, use_batch_norm=True, is_last=False):
    layers = []
    layers.append(nn.Conv2d(channels_in, channels_out, kernel, stride, padding))
    
    if is_last:
        layers.append(nn.Sigmoid())
    elif not use_batch_norm:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        layers.append(nn.BatchNorm2d(channels_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, ndf, nchannels=1):
        super().__init__()

        self.model = nn.Sequential(
            # Convolution (32 filters, kernel 4, stride 2, padding 1, no bias) + Batch Norm + LeakyReLU (α = 0.2)
            get_downscaling_block(nchannels, ndf, kernel=4, stride=2, padding=1, use_batch_norm=True),
            # Convolution (64 filters, kernel 4, stride 2, padding 1, no bias) + Batch Norm + LeakyReLU (α = 0.2)
            get_downscaling_block(ndf, ndf*2, kernel=4, stride=2, padding=1, use_batch_norm=True),
            # Convolution (128 filters, kernel 4, stride 2, padding 1, no bias) + Batch Norm + LeakyReLU (α = 0.2)
            get_downscaling_block(ndf*2, ndf*4, kernel=4, stride=2, padding=1, use_batch_norm=True),
            # Convolution (1 filter, kernel 4, stride 1, padding 0, no bias) + Sigmoid activation
            get_downscaling_block(ndf*4, 1, kernel=4, stride=1, padding=0, is_last=True, use_batch_norm=False),
        )


    def forward(self, x):
        return self.model(x).squeeze(1).squeeze(1) # remove spatial dimensions --> TODO: it can be done with Unflatten as in the generator
    
