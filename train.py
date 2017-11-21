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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--jcbSize', type=int, default=8, help='size of sub-dimension for computing jacobian')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=32, help='dimension of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='input channel')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate of discriminator, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.001, help='learning rate of generator, default=0.001')
parser.add_argument('--lrDecay', type=float, default=0.95, help='learning rate decay')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=20, help='weight for reconstruction')
parser.add_argument('--beta', type=float, default=0.01, help='weight for orthogonal loss')
parser.add_argument('--theta', type=float, default=0.1, help='weight for adversarial loss of recontructed images')
parser.add_argument('--gamma', type=float, default=1.)
parser.add_argument('--delta', type=float, default=0.0001, help='step size for computing jacobian')
parser.add_argument('--var', type=float, default=3, help='variance of gaussian noise')

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
                                   transforms.Scale(opt.imageSize+10),
                                   transforms.RandomCrop(opt.imageSize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize+10),
                            transforms.RandomCrop(opt.imageSize),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize+10),
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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
alpha = float(opt.alpha)
jcbSize = int(opt.jcbSize)

lrD = float(opt.lrD)
lrG = float(opt.lrG)

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

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        if opt.imageSize == 64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif opt.imageSize == 32:
            self.main = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_L1 = nn.L1Loss()

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
input_tile = torch.FloatTensor(opt.batchSize*jcbSize, nc, opt.imageSize, opt.imageSize)
regress_img = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
pos_noise = torch.FloatTensor(opt.batchSize*jcbSize, nz, 1, 1)
zero_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, opt.var)
label = torch.FloatTensor(opt.batchSize)
eye_label = torch.FloatTensor(opt.batchSize, jcbSize, jcbSize)
eye_nz = torch.FloatTensor(opt.batchSize, jcbSize, nz)

real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterion_L1.cuda()
    input, label = input.cuda(), label.cuda()
    input_tile = input_tile.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    zero_noise = zero_noise.cuda()
    pos_noise = pos_noise.cuda()
    regress_img = regress_img.cuda()
    eye_label = eye_label.cuda()
    eye_nz = eye_nz.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: minimize -log(D(x)) - log(1 - D(G(x,z))) - \theta * log(1 - D(G(x,0)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        regress_img.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        labelv = Variable(label)
        regress_imgv = Variable(regress_img)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, opt.var)
        noisev = Variable(noise)
        G_x_z = netG(inputv, noisev)
        
        zero_noise.resize_(batch_size, nz, 1, 1).fill_(0)
        zero_noisev = Variable(zero_noise)
        G_x_0 = netG(inputv, zero_noisev)        
        
        labelv = Variable(label.fill_(fake_label))
        output = netD(G_x_z.detach())
        output_0 = netD(G_x_0.detach())
        #Treat G_x_z and G_x_0 as fake
        errD_fake = criterion(output, labelv) + opt.theta*criterion(output_0, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: minimize -log(D(G(x,z))) - \theta * log(D(G(x,0))) + \alpha * L1(x, G(x,0)) + \beta * L1(JxJx^T, I)
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(G_x_z)
        output_0 = netD(G_x_0)
        errG = criterion(output, labelv) + opt.theta*criterion(output_0, labelv)
        errL1 = criterion_L1(G_x_0, regress_imgv)
        
        #Jacobian
        input_tile.resize_(batch_size*jcbSize, nc, opt.imageSize, opt.imageSize)
        real_cpu_tile = real_cpu.repeat(jcbSize, 1, 1, 1, 1)
        real_cpu_tile = real_cpu_tile.transpose(0, 1).contiguous()
        real_cpu_tile = real_cpu_tile.view(batch_size*jcbSize, nc, opt.imageSize, opt.imageSize)
        input_tile.copy_(real_cpu_tile)
        input_tilev = Variable(input_tile)
        
        eye_label.resize_(batch_size, jcbSize, jcbSize).copy_(torch.eye(jcbSize), broadcast = True)
        eye_labelv = Variable(eye_label)
        
        eye_nz.resize_(batch_size, jcbSize, nz).copy_(torch.eye(nz)[torch.randperm(nz)[:jcbSize]], broadcast = True)        
        pos_noise_flatten = (opt.delta * eye_nz).view(batch_size*jcbSize, nz, 1, 1)
        pos_noise.resize_(batch_size*jcbSize, nz, 1, 1).copy_(pos_noise_flatten)
        pos_noisev = Variable(pos_noise)
        
        Jx = (netG(input_tilev, pos_noisev) - netG(input_tilev, -pos_noisev))/(2*opt.delta)
        Jx = Jx.view(batch_size, jcbSize, -1)
        Jx_T = Jx.transpose(1, 2)

        errOrth = criterion_L1(torch.matmul(Jx,Jx_T), opt.gamma*eye_labelv)
        
        err = errG + errL1*alpha + errOrth*opt.beta
        err.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f L1_Loss: %.4f Orth_Loss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], errL1.data[0], errOrth.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            netG.eval()
            vutils.save_image(inputv.data,
                    '%s/real_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
            fake = netG(inputv, fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
            recons = netG(inputv, zero_noisev)
            vutils.save_image(recons.data,
                    '%s/reconstruction_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)
            netG.train(True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    # update learning rate
    lrD = lrD * opt.lrDecay
    lrG = lrG * opt.lrDecay
    
    for param_group in optimizerD.param_groups:
        param_group['lr'] = lrD
        
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lrG