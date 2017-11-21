from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import functools


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--naxis', type=int, default=5, help='interpolation axis')
parser.add_argument('--testSize', type=int, default=20, help='number of test images')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='input channel')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--var', type=float, default=3)
parser.add_argument('--delta', type=float, default=0.0001)

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.RandomCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.RandomCrop(opt.imageSize),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.RandomCrop(opt.imageSize),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.RandomCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
nc = int(opt.nc)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#        m.weight.data.normal_(0.0, 0.02)
        init.xavier_uniform(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, nc, ngf=64, nz=100,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(_netG, self).__init__()
        
        if opt.imageSize == 64:
            self.encoder = nn.Sequential(
                # input is X, going into a convolution
                nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, True),
                # state size, (ngf) x 32 x 32
                nn.Conv2d(ngf, 2 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(2 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (2*ngf) x 16 x 16
                nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(4 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (4*ngf) x 8 x 8
                nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(8 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (8*ngf) x 4 x 4
                nn.Conv2d(8 * ngf, nz, 4, 1, 0, bias=False),
                nn.BatchNorm2d(nz)
                # state size, (8*ngf) x 1 x 1
            )
            self.decoder = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(nz, 8 * ngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(8 * ngf),
                # state size, (8*ngf) x 4 x 4
                nn.ReLU(True),
                nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(4 * ngf),
                # state size, (4*ngf) x 8 x 8
                nn.ReLU(True),
                nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(2 * ngf),
                # state size, (2*ngf) x 16 x 16
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                # state size, (ngf) x 32 x 32
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size, (nc) x 64 x 64    
            )
        elif opt.imageSize == 32:
            self.encoder = nn.Sequential(
                # input is X, going into a convolution
                nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, True),
                # state size, (ngf) x 16 x 16
                nn.Conv2d(ngf, 2 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(2 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (2*ngf) x 8 x 8
                nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(4 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (4*ngf) x 4 x 4
                nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(8 * ngf),
                nn.LeakyReLU(0.2, True),
                # state size, (8*ngf) x 2 x 2
                nn.Conv2d(8 * ngf, nz, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nz)
                # state size, (nz) x 1 x 1
            )
            self.decoder = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(nz, 8 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(8 * ngf),
                # state size, (8*ngf) x 2 x 2
                nn.ReLU(True),
                nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(4 * ngf),
                # state size, (4*ngf) x 4 x 4
                nn.ReLU(True),
                nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(2 * ngf),
                # state size, (2*ngf) x 8 x 8
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                # state size, (ngf) x 16 x 16
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size, (nc) x 32 x 32    
            )

    def forward(self, x, z):
        output = self.encoder(x)
        output = output + z
        output = self.decoder(output)
        
        return output

if opt.cuda:
    netG = torch.nn.DataParallel(_netG(nc,ngf,nz), device_ids=range(ngpu))
else:
    netG = _netG(nc, ngf, nz)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netG.eval()

input = torch.FloatTensor(1, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(1, nz, 1, 1).fill_(0)
noise_ = torch.FloatTensor(1, nz, 1, 1)

random_axis = [i for i in range(nz)]
random.shuffle(random_axis)
random_axis = random_axis[:opt.naxis]

if opt.cuda:
    netG.cuda()
    input = input.cuda()
    noise = noise.cuda()
    noise_ = noise_.cuda()
    
for i, data in enumerate(dataloader, 0):
    real_cpu, _ = data
    if opt.cuda:
        real_cpu = real_cpu.cuda()
    input.copy_(real_cpu)
    inputv = Variable(input,volatile=True)
    
    if i < opt.testSize:
        vutils.save_image(inputv.data,
                    '%s/real_samples_%03d.png' % (opt.outf, i),
                    normalize=True)
        
        noise_.copy_(noise)
        noise_[0][random_axis[0]].fill_(-5*opt.var)
        noise_v = Variable(noise_,volatile=True)
        fake_panel = netG(inputv, noise_v)
        for ii in range(0, opt.naxis):
            noise_.copy_(noise)
            if ii == 0:
                for jj in range(1, 15):
                    noise_[0][random_axis[ii]].fill_(-5*opt.var+5./7.*opt.var*jj)
                    noise_v = Variable(noise_,volatile=True)
                    fake = netG(inputv, noise_v)
                    fake_panel = torch.cat((fake_panel, fake), 0)
            else:
                for jj in range(0, 15):
                    noise_[0][random_axis[ii]].fill_(-5*opt.var+5./7.*opt.var*jj)
                    noise_v = Variable(noise_,volatile=True)
                    fake = netG(inputv, noise_v)
                    fake_panel = torch.cat((fake_panel, fake), 0)                                         

        vutils.save_image(fake_panel.data,
                    '%s/fake_samples_%03d.png' % (opt.outf, i),
                    normalize=True, nrow=15)
    else:
        break