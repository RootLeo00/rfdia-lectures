import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.utils
from torchvision.utils import make_grid

import torchvision.datasets

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 128 # Images per batch

# Resize to 32x32 for easier upsampling/downsampling
mytransform = transforms.Compose([transforms.Resize(32),
                                  transforms.ToTensor(),
                                 transforms.Normalize((.5), (.5))]) # normalize between [-1, 1] with tanh activation

mnist_train = torchvision.datasets.MNIST(root='.', download=True, transform=mytransform)

dataloader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True)

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

class ConditionalGenerator(nn.Module):
    def __init__(self, nz, nc, ngf, nchannels=1):
        super().__init__()

        # Transpose Convolution (256 filters, kernel 4, stride 1, padding 0, no bias) + Batch Norm + ReLU
        self.upscaling_z = get_upscaling_block(nz, ngf*16, kernel=4, stride=1, padding=0) # branch 1
        self.upscaling_c = get_upscaling_block(nc, ngf*16, kernel=4, stride=1, padding=0) # branch 2

        self.rest_model = nn.Sequential(
            # Concatenate branches 1 and 2 (256 + 256 channels)
            get_upscaling_block(ngf*16 + ngf*16, ngf*16, kernel=4, stride=2, padding=1),
            # Transpose Convolution (128 filters, kernel 4, stride 2, padding 1, no bias) + Batch Norm + ReLU
            get_upscaling_block(ngf*16, ngf*8, kernel=4, stride=2, padding=1),
            # Transpose Convolution (1 filter, kernel 4, stride 2, padding 1, no bias)
            get_upscaling_block(ngf*8, nchannels, kernel=4, stride=2, padding=1, last_layer=True)
        )

    def forward(self, x, y):
        x = x.unsqueeze(2).unsqueeze(2)
        y = y.unsqueeze(2).unsqueeze(2)
        
        x = self.upscaling_z(x)
        y = self.upscaling_c(y)
        
        combined = torch.cat([x, y], 1)
        return self.rest_model(combined)
    

# Discriminator

'''
The conditional discriminator needs the label information as well as the latent vector. We will combine the latent vector and the class information in the following way:

- The class information for the discriminator will be represented as a one-hot vector sized `[batch_size, 10]` (since there are 10 classes in MNIST)
- The latent vector for the generator will still be sized `[batch_size, nz]`

1. Transform both of these modalities into 'images' (by adding dimensions)
2. Like before, apply the first upscaling block to both of these 'images'. We will now have 2 separate blocks sized

'''

class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf, nc, nchannels=1):
        super().__init__()
        self.downscale_x = get_downscaling_block(nchannels, ndf*2, 4, 2, 1, use_batch_norm=False)
        self.downscale_y = get_downscaling_block(nc, ndf*2, 4, 2, 1, use_batch_norm=False)


        self.rest = nn.Sequential(
            get_downscaling_block(ndf*2 + ndf*2, ndf*8, kernel=4, stride=2, padding=1, use_batch_norm=True),
            get_downscaling_block(ndf*8, ndf*16, kernel=4, stride=2, padding=1, use_batch_norm=True),
            get_downscaling_block(ndf*16, 1, kernel=4, stride=1, padding=0, is_last=True, use_batch_norm=False),
        )


    def forward(self, x, y):
        #Expansion of y into a tensor sized 10 × 32 × 32
        y = y.unsqueeze(2).unsqueeze(2).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.downscale_x(x)
        y = self.downscale_y(y)
        # Concatenate the two feature maps along the channel dimension
        combined = torch.cat([x, y], dim=1)

        return self.rest(combined).squeeze(1).squeeze(1) # remove spatial dimensions


def sample_z(batch_size, nz):
    return torch.randn(batch_size, nz, device=device)

# this is for the real ground-truth label
def get_labels_one(batch_size):
    r = torch.ones(batch_size, 1)
    return r.to(device)

# this is for the generated ground-truth label
def get_labels_zero(batch_size):
    r = torch.zeros(batch_size, 1)
    return r.to(device)


# To initialize the weights of a GAN, the DCGAN paper found that best results are obtained
# with Gaussian initialization with mean=0; std=0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# for visualization
to_pil = transforms.ToPILImage()
renorm = transforms.Normalize((-1.), (2.))


nz = 100
ndf = 32
ngf = 32
nchannels= 1
lr_d = 0.0002
lr_g = 0.0005
beta1= 0.5
display_freq = 200

nc= 10

netD = ConditionalDiscriminator(ndf, nc, nchannels=1).to(device)
netG = ConditionalGenerator(nz, nc, ngf).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

g_opt = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
d_opt = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))


nb_epochs = 5

g_losses = []
d_losses = []
criterion = nn.BCELoss() # we will build off of this to make our final GAN loss!


j = 0

z_test = sample_z(100, nz)  # we generate the noise only once for testing

for epoch in range(nb_epochs):
    pbar = tqdm(enumerate(dataloader))
    for i, batch in pbar:
        im, labels = batch
        im = im.to(device)
        y = F.one_hot(labels).float().to(device)
        cur_batch_size = im.shape[0]
        z = sample_z(cur_batch_size, nz)
        # label_real = torch.full((cur_batch_size,), 1., dtype=torch.float, device=device)
        label_real= get_labels_one(cur_batch_size).view(-1)

        # 2. forward pass through D (=Classify real image with D)
        yhat_real = netD(im,y).view(-1) # the size -1 is inferred from other dimensions

        # 3. forward pass through G (=Generate fake image batch with G)
        y_fake = netG(z, y)
        # label_fake=label_real.fill_(0.)
        label_fake= get_labels_zero(cur_batch_size).view(-1)

        # 4. Classify fake image with D
        yhat_fake = netD(y_fake.detach(), y).view(-1)

        ### Discriminator
        d_loss = criterion(yhat_real,label_real) + criterion(yhat_fake,label_fake) # TODO check loss
        d_opt.zero_grad()
        d_loss.backward(retain_graph=True) # we need to retain graph=True to be able to calculate the gradient in the g backprop
        d_opt.step()


        ### Generator
        # Since we just updated D, perform another forward pass of all-fake batch through D
        yhat_fake = netD(y_fake, y).view(-1)
        g_loss = criterion(yhat_fake, label_real) # fake labels are real for generator cost
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        # Save Metrics
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        avg_real_score = yhat_real.mean().item()
        avg_fake_score = yhat_fake.mean().item()

        pbar.set_description(f"it: {j}; g_loss: {g_loss}; d_loss: {d_loss}; avg_real_score: {avg_real_score}; avg_fake_score: {avg_fake_score}")

        if i % display_freq == 0:
            labels = torch.arange(0, 10).expand(size=(10, 10)).flatten().to(device)
            y = F.one_hot(labels).float().to(device)
            fake_im = netG(z_test, y)

            un_norm = renorm(fake_im) # for visualization

            grid = torchvision.utils.make_grid(un_norm, nrow=10)
            pil_grid = to_pil(grid)

            plt.imshow(pil_grid)
            # plt.show()
            plt.savefig("./results/generated_img_cgan.png")
            plt.clf()


            plt.plot(range(len(g_losses)), g_losses, label='g loss')
            plt.plot(range(len(g_losses)), d_losses, label='d loss')

            plt.legend()
            # plt.show()
            plt.savefig("./results/losses_cgan.png")
            plt.clf()


        j += 1
