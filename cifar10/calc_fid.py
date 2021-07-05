"""
Source code for training Run-Sort-ReRun on CIFAR-10 

Adapted from   https://github.com/Yuantian013/E2GAN

Auxiliary code to compute FID of pre-traiend model
"""


import sys, os
import torch
import numpy as np
import imageio
import torch.nn as nn
# from model import Generator
from torchvision import utils
import cfg
import models_search


img_size = 32
latent_size = 128


args = cfg.parse_args()

args.img_size = 32
args.bottom_width = 4
args.gen_model = 'shared_gan_leaky'
args.latent_dim = 128
args.gf_dim = 256
args.g_spectral_norm = False
# args.load_path = 'checkpoints/e2gan_cifar.pth'
args.load_path = 'checkpoints/RSR_CIFAR10_pretrained.pth' # download from https://www.dropbox.com/s/9vluprfcxuhygpi/RSR_CIFAR10_pretrained.pth?dl=0

args.arch = [0, 1, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2] # e2gan architecture is defined this way (see paper)

G = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
G.set_arch(args.arch, cur_stage=2)

# load weights
checkpoint_file = args.load_path
assert os.path.exists(checkpoint_file)
checkpoint = torch.load(checkpoint_file)

if 'avg_gen_state_dict' in checkpoint:
   G.load_state_dict(checkpoint['avg_gen_state_dict'])
   epoch = checkpoint['epoch'] - 1
   print(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
else:
   G.load_state_dict(checkpoint)
   print(f'=> loaded checkpoint {checkpoint_file}')

batch_size = 16

outdir_images = 'outimgs/fakes/2880/'

if not os.path.exists(outdir_images):
    os.mkdir(outdir_images)

count_imgs = 0
G.eval()
with torch.no_grad():
    Nb = 11000//batch_size
    for i in range(Nb):
        print('processing batch %i of %i' % (i, Nb))
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images = G(z)

        for i in range(batch_size):
            utils.save_image(
                fake_images[i],
                '%s/img_%06i.png' % (outdir_images, count_imgs),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            count_imgs +=1

fid_command = 'python ../pytorch-fid/pytorch_fid/fid_score.py %s fid_stats_cifar10_train.npz  --device cuda:0 ' % outdir_images
os.system(fid_command)
