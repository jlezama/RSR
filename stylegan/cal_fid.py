import sys, os
import torch
import numpy as np
import imageio
import torch.nn as nn
from model import Generator
from torchvision import utils

img_size = 256
latent_size = 512
load_path = 'weights/G_001100.pth'
# load_path = 'stylegan2-church-config-f.pt'

G = Generator(img_size, latent_size, 8, channel_multiplier=2).to("cuda")
G = nn.DataParallel(G)
ckpt = torch.load(load_path, map_location=lambda storage, loc: storage)
# G.load_state_dict(ckpt["g_ema"])
G.load_state_dict(ckpt)

batch_size = 16

outdir_images = 'outimgs/fakes/1900/'

if not os.path.exists(outdir_images):
    os.mkdir(outdir_images)

count_imgs = 0
G.eval()
with torch.no_grad():
    Nb = 11000//batch_size
    for i in range(Nb):
        print('processing batch %i of %i' % (i, Nb))
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images, _ = G([z])

        # fake_images_np = fake_images.cpu().detach().numpy()

        # fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, img_size, img_size)
        # fake_images_np = ((fake_images_np.transpose((0, 2, 3, 1) ) /2.0 + .5 ) *255).astype(np.uint8)

        for i in range(batch_size):
            # imageio.imwrite('%s/img_%06i.png' % (outdir_images, count_imgs), fake_images_np[i])
            utils.save_image(
                fake_images[i],
                '%s/img_%06i.png' % (outdir_images, count_imgs),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            count_imgs +=1

fid_command = 'python ../pytorch-fid/pytorch_fid/fid_score.py %s fid_stats_lsun_church.npz  --device cuda:0 ' % outdir_images
os.system(fid_command)
