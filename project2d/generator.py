import torch.nn as nn

def get_upscaling_block(channels_in, channels_out, kernel, stride, padding, last_layer=False):
    '''
    Each transpose conv will be followed by BatchNorm and ReLU,
    except the last block (which is only followed by tanh)
    '''
    layers = []
    layers.append(nn.ConvTranspose2d(channels_in, channels_out, kernel, stride, padding))
    if not last_layer:
        layers.append(nn.BatchNorm2d(channels_out))
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nchannels=1):
        '''
        nz: The latent size (100 in our case)
        ngf: The channel-size before the last layer (32 our case)
        '''
        super().__init__()


        self.model = nn.Sequential(
            # Reshape z [128,1000] into [n_batch, n_z, 1, 1]
            nn.Unflatten(dim=1, unflattened_size=(nz, 1, 1)),
            
            # First upscaling block -> [n_batch, ngf*4, 4, 4]
            get_upscaling_block(nz, ngf*4, kernel=4, stride=1, padding=0),
            
            # Second upscaling block -> [n_batch, ngf*2, 8, 8]
            get_upscaling_block(ngf*4, ngf*2, kernel=4, stride=2, padding=1),
            
            # Third upscaling block -> [n_batch, ngf, 16, 16]
            get_upscaling_block(ngf*2, ngf, kernel=4, stride=2, padding=1),
            
            # Fourth (and last) upscaling block -> [n_batch, 1, 32, 32]
            get_upscaling_block(ngf, nchannels, kernel=4, stride=2, padding=1, last_layer=True)
        )

    def forward(self, z):
        # print(z.shape)
        # Done in the nn.Sequential model with Unflatten 
        # x = z.unsqueeze(2).unsqueeze(2) # give spatial dimensions to z
        # print(x.shape)
        return self.model(z)
