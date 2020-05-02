import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs",type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality z of the latent space")
parser.add_argument("--ngf", type=int, default=64, help="generator feature map number")
parser.add_argument("--ndf", type=int, default=64, help="discriminator feature map number")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")


opt = parser.parse_args()
print(opt)
os.makedirs("images",exist_ok=True)
image_shape = (opt.channels, opt.img_size,opt.img_size)

cuda = True if  torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # 100*1*1
            nn.ConvTranspose2d(opt.latent_dim, opt.ngf*8,4,1,0),
            nn.BatchNorm2d(opt.ngf*8),
            nn.ReLU(True),
            #64*8*4*4
            nn.ConvTranspose2d( opt.ngf*8, opt.ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(opt.ngf*4),
            nn.ReLU(True),
            # 64*4*8*8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # 64*2*16*16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 1),
            nn.ReLU(True),
            # 64*32*32
            nn.ConvTranspose2d(opt.ngf * 1,1,4,2,1),
            nn.Tanh()
            # 1*64*64
        )
    def forward(self, x):
        img = self.model(x)
        print(img.size())
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            #1*64*=64 (w-k+2*padding)/stride +1
            nn.Conv2d(1,opt.ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            ## 32
            nn.Conv2d(opt.ndf, opt.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            ## 16
            nn.Conv2d(opt.ndf*2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ## 8

            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            ## 4
            nn.Conv2d(opt.ndf * 8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.model(x)
        return x


dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
adversarial_loss = torch.nn.BCELoss()

generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
os.makedirs("data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, label) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).view(-1, opt.latent_dim, 1, 1))

        # Generate a batch of images
        gen_imgs = generator(z)
        print(z.size())
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)



