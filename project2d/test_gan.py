import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from discriminator import Discriminator
from generator import Generator
from utils import *

plt.rcParams["figure.figsize"] = (10, 6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import other necessary libraries and classes...

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="GAN Training Script")
    parser.add_argument("--ngf", type=int, default=32, help="Channel size before the last layer in Generator")
    parser.add_argument("--ndf", type=int, default=32, help="Channel size in Discriminator")
    parser.add_argument("--init-type", choices=["custom", "pytorch"], default="pytorch",
                        help="Weight initialization type")
    parser.add_argument("--loss-type", choices=["true", "default"], default="default",
                        help="Type of training loss for the generator")
    parser.add_argument("--lr-d", type=float, default=0.0002, help="Learning rate for the discriminator")
    parser.add_argument("--lr-g", type=float, default=0.0005, help="Learning rate for the generator")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--nz", type=int, default=100, help="Latent size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--nchannels", type=int, default=1, help="Number of channels for inputs of Discriminator")
    parser.add_argument("--display-freq", type=int, default=200, help="Display frequency of images")
    parser.add_argument("--savepath", default="", help="Path to save images on disk")
    
    return parser.parse_args()

def main(args):
    nz = args.nz
    ngf = args.ngf
    ndf = args.ndf
    lr_d = args.lr_d
    lr_g = args.lr_g
    beta1 = args.beta1
    nb_epochs = args.epochs
    nchannels=args.nchannels
    batch_size=args.batch_size
    display_freq=args.display_freq
    savepath=args.savepath

    # Use args.init_type to decide whether to use custom weight initialization
    if args.init_type == "custom":
        netD.apply(weights_init)
        netG.apply(weights_init)

    # loading data
    # Resize to 32x32 for easier upsampling/downsampling
    mytransform = transforms.Compose([transforms.Resize(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize((.5), (.5))]) # normalize between [-1, 1] with tanh activation

    mnist_train = torchvision.datasets.MNIST(root='.', download=True, transform=mytransform)

    dataloader = DataLoader(dataset=mnist_train,
                            batch_size=batch_size,
                            shuffle=True)
        
    # for visualization
    to_pil = transforms.ToPILImage()
    renorm = transforms.Normalize((-1.), (2.))

    netD = Discriminator(ndf, nchannels).to(device)
    netG = Generator(nz, ngf).to(device)

    netD.apply(weights_init)
    netG.apply(weights_init)

    g_opt = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
    d_opt = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))

    criterion = nn.BCELoss() # we will build off of this to make our final GAN loss!

    g_losses = []
    d_losses = []

    j = 0

    z_test = sample_z(64, nz) # we generate the noise only once for testing

    for epoch in range(nb_epochs):

        # train
        pbar = tqdm(enumerate(dataloader))
        for i, batch in pbar:

            # 1. construct a batch of real samples from the training set (sample a z vector)
            im, _ = batch # we don't care about the label for unconditional generation
            im = im.to(device) # real image
            cur_batch_size = im.shape[0] # batch size
            z = sample_z(cur_batch_size, nz)
            # label_real = torch.full((cur_batch_size,), 1., dtype=torch.float, device=device)
            label_real= get_labels_one(cur_batch_size).view(-1)

            # 2. forward pass through D (=Classify real image with D)
            yhat_real = netD(im).view(-1) # the size -1 is inferred from other dimensions

            # 3. forward pass through G (=Generate fake image batch with G)
            y_fake = netG(z)
            # label_fake=label_real.fill_(0.)
            label_fake= get_labels_zero(cur_batch_size).view(-1)

            # 4. Classify fake image with D
            yhat_fake = netD(y_fake.detach()).view(-1) # detach() in order to stop gradient computation until y_fake (to fasten training)

            ### Discriminator
            d_loss = criterion(yhat_real,label_real) + criterion(yhat_fake,label_fake) # TODO check loss
            d_opt.zero_grad()
            d_loss.backward(retain_graph=True) # we need to retain graph=True to be able to calculate the gradient in the g backprop
            d_opt.step()


            ### Generator
            # Since we just updated D, perform another forward pass of all-fake batch through D
            yhat_fake = netD(y_fake).view(-1)
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
                
                # print(z_test.shape)
                # print(netG.model)
                fake_im = netG(z_test)

                un_norm = renorm(fake_im) # for visualization

                grid = torchvision.utils.make_grid(un_norm, nrow=8)
                pil_grid = to_pil(grid)

                # print("generated images")
                plt.imshow(pil_grid)
                # plt.show()
                plt.savefig("./results/generated_img_gan1_"+savepath+".png")
                plt.clf()

                plt.plot(range(len(g_losses)), g_losses, label='g loss')
                plt.plot(range(len(g_losses)), d_losses, label='d loss')

                plt.legend()
                # plt.show()
                plt.savefig("./results/losses_gan1_"+savepath+".png")
                plt.clf()

            j += 1

if __name__ == "__main__":
    args = get_args()
    main(args)
