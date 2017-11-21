# Localized Generative Adversarial Nets (LGAN) for Image Generation with Diversity
Author:Liheng Zhang, Date: 11/21/2017

This is the project for the following technical report:

**Guo-Jun Qi, Liheng Zhang, Hao Hu. Global versus Localized Generative Adversarial Nets. arXiv: 1711.06020 [[pdf]     (https://arxiv.org/pdf/1711.06020.pdf)]**

Questions about the source codes can be directed to Liheng Zhang at lihengzhang1993@knights.ucf.edu.

## Requirements
- Python == 2.7
- Pytorch == 0.2.0_4

## For celebA dataset
1. Setup and download dataset
```bash
mkdir celebA; cd celebA
```
Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".
```bash
unzip img_align_celeba.zip; cd ..
```

2. Train LGAN
```bash
python train.py --dataset folder --dataroot ./celebA --imageSize 64 --nz 32 --nc 3 --cuda --outf "./results/celebA"
```
3. Densely sample images with diversity
```bash
python dense_sample.py --dataset folder --dataroot ./celebA --imageSize 64 --nz 32 --nc 3 --cuda --netG "./results/celebA/netG_epoch_24.pth" --outf "./results/celebA" 
```
## For mnist dataset
1. Train LGAN
```bash
python train.py --dataset mnist --dataroot ./mnist --imageSize 32 --nz 10 --nc 1 --lrD 0.0001 --lrG 0.0005 --cuda --outf "./results/mnist"
```
2. Densely sample images with diversity
```bash
python dense_sample.py --dataset mnist --dataroot ./mnist --imagesSize 32 --nz 10 --nc 1 --cuda --netG "./results/mnist/netG_epoch_24.pth" --outf "./results/mnist"
```
## Acknowledge
Parts of codes are reused from DCGAN at https://github.com/pytorch/examples/tree/master/dcgan
