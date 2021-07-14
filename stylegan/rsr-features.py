# -*- coding: utf-8 -*-
"""
RSR Algorithm by Jose Lezama <jlezama@fing.edu.uy>

Adapted from StyleGAN2
"""

import sys, os
from torchvision import utils

import time
import cfg
from model import Generator
from dataset import MultiResolutionDataset
sys.path.append('pytorch-fid/pytorch_fid')

from inception import InceptionV3
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
inception_model = InceptionV3([block_idx], normalize_input=False)
inception_model.cuda()
inception_model.eval()

def preprocess_for_inception(img):
   #return (img+1.0)/2.0
   return img

# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.datasets as dset
import numpy as np

num_gpu = 1 if torch.cuda.is_available() else 0

# load the StyleGAN2 models
args = cfg.parse_args()

args.img_size = 256
args.latent_dim = 512
args.load_path = 'stylegan2-church-config-f.pt'
n_mlp = 8
channel_multiplier = 2
device = "cuda"

G = Generator(args.img_size, args.latent_dim, n_mlp, channel_multiplier=channel_multiplier).to(device)

# load weights
ckpt = torch.load(args.load_path, map_location=lambda storage, loc: storage)
G.load_state_dict(ckpt["g_ema"])
G = nn.DataParallel(G)

print(G)

G.train()

description='full_lbs_%i_%i' % (args.lbs_init, args.lbs_end)

print('running experiment %s' % description)

outdir_images = 'outimgs/fakes/%s/' % (description)
outdir_weights = 'weights/%s/' % (description)

os.system('mkdir -p %s' % outdir_images)
os.system('mkdir -p %s' % outdir_weights)

batch_size =  args.gen_batch_size
latent_size = args.latent_dim

small_batch_size = batch_size
large_batch_size = args.lbs_init

large_batch_size_init = large_batch_size
large_batch_size_end =  args.lbs_end + batch_size

max_epoch = args.max_epoch

Nsteps = (large_batch_size_end-large_batch_size_init)//small_batch_size
step_length = max_epoch//Nsteps

NS = large_batch_size//small_batch_size


N_rotmat = 4000 # number of random projections
d_img = 2048 # inception feature dimension

G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, weight_decay=0) # set LR and weight decay here
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_opt, max_epoch)

if args.dataset == "LSUN_bedrooms":
   dataset = dset.LSUN(root='.data/',
                       classes=['bedroom_train'],
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == "LSUN_churches":
   transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
   args.data_path = '.data/church'
   dataset = MultiResolutionDataset(args.data_path, transform, args.img_size)
elif args.dataset == "celeba":
   dataset = dset.CelebA(root='.data/', download=True,
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == "imagenet":
   dataset = dset.ImageNet(root='.data/', download=True,
                        transform=transforms.Compose([
                           # transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
else:
   dataset = dset.CIFAR10(root='.data/', download=True,
                        transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
nc=3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=small_batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)

dataloader_iterator = iter(dataloader)


losses = []

for epoch in range(max_epoch):

  scheduler.step() 
   
  new_large_batch_size = large_batch_size_init + (epoch//step_length)*small_batch_size

  new_lbs = 0
     
  if epoch==0: # first iteration uses random projections
     rotmat_img = torch.randn(d_img, N_rotmat).cuda()
     rotmat_img = rotmat_img/torch.sqrt(torch.sum(rotmat_img**2, dim=0))

  elif epoch >0: # following iterations uses pairs and worst projections from previous iteration
   with torch.no_grad():
     pworst = 1/3.0
     # keep 1/3rd of worst projections, add 2/3rd new ones
     worst_values_img, worst_index_img = torch.sort(G_loss_all_img, descending=True)
     rotmat_img_prev = rotmat_img[:,worst_index_img[:int(N_rotmat*pworst)]]

     # rotmatimg will be taken from pairs of gt, output
     N_rotmat_new = int(N_rotmat*(1-pworst))
     ix_gt = np.random.randint(0,large_batch_size - small_batch_size*new_lbs, N_rotmat_new)
     ix_output = np.random.randint(0,large_batch_size - small_batch_size*new_lbs, N_rotmat_new)

     vectors_gt = all_gt[ix_gt, :].detach().t().cuda()
     vectors_out = all_output_img[ix_output, :].detach().t()

     rotmat_img = (vectors_gt-vectors_out)

     # normalize
     rotmat_img = rotmat_img/torch.sqrt(torch.sum(rotmat_img**2, dim=0))
     rotmat_img = torch.cat((rotmat_img, rotmat_img_prev), dim=1)

  # initialize tensors for noise vectors, real data and fake data
  all_z = torch.randn(large_batch_size, latent_size).cuda()
  all_gt = torch.zeros(large_batch_size, d_img).cuda()
  all_output_img = torch.zeros(large_batch_size, d_img).cuda()

  gt_images = [] # auxiliary list for use_gt = True
  fake_images_debug = None


  ####################################################
  # STEP 1. RUN
  with torch.no_grad():
    for idx in range(NS):

      if epoch>=0: 
          try:
             images = next(dataloader_iterator)
          except:
             dataloader_iterator = iter(dataloader)
             images = next(dataloader_iterator)
          images = images.cuda()
          inception_features_gt = inception_model(images)[0].view(batch_size, -1)
          all_gt[idx*batch_size:(idx+1)*batch_size,:] = inception_features_gt
    
      z = all_z[idx*batch_size:(idx+1)*batch_size,:]

      fake_images, _ = G([z], randomize_noise=False)

      # compute inception feature
      inception_features = inception_model(preprocess_for_inception(fake_images))[0].view(batch_size, -1)

      all_output_img[idx*batch_size:(idx+1)*batch_size,:] = inception_features

  ## finished computing features, now project
  with torch.no_grad():
     all_output_img_projected = all_output_img.mm(rotmat_img)
     all_gt_projected = all_gt.mm(rotmat_img)

  ####################################################
  # STEP 2. SORT
  with torch.no_grad():
     # move to cpu, sort, move back to gpu
     [_, out_img_sort_ix] = torch.sort(all_output_img_projected.cpu(), dim=0)
     out_img_sort_relative = out_img_sort_ix.argsort(0)
     out_img_sort_relative = out_img_sort_relative.cuda()

     [gt_sort_val, _] = torch.sort(all_gt_projected.cpu(), dim=0)
     gt_sort_val = gt_sort_val.cuda()

  
  ####################################################
  # STEP 3. RE-RUN
  # initialize gradient                                                                      
  G_opt.zero_grad()
  full_batch_loss = 0

  G_loss_all_feat = 0
  G_loss_all_img = 0

  SQRT2 = 1.4142135623731

  # now do actual comparison 
  for idx in range(0,large_batch_size,small_batch_size):
      z = all_z[idx:idx+small_batch_size,:]

      fake_images, _ = G([z], randomize_noise=False)

      # compute inception feature
      inception_features = inception_model(preprocess_for_inception(fake_images))[0].view(batch_size, -1)
      output_img = inception_features.mm(rotmat_img) # project

      # get the relative position of the output
      rel_ix_img = out_img_sort_relative[idx:idx+small_batch_size,:]

      # now get the equivalent positions of the gt
      gt = gt_sort_val.gather(0, rel_ix_img).cuda()

      diff_img = (gt-output_img)**2

      threshold_img = 1e-4 # don't penalize too small differences, this is normal even for samples of the same distribution. Trying other values for this hyperparameter might be interesing
      diff_img = (torch.clamp(diff_img, min=threshold_img)-threshold_img)

      G_loss_row_img = torch.sum(diff_img, dim=0) / large_batch_size 
      G_loss_img = torch.sum(G_loss_row_img)/ rotmat_img.shape[1]

      G_loss =  G_loss_img
      G_loss.backward()
      G_loss_all_img += G_loss_row_img.detach().cpu()
      full_batch_loss += G_loss.item()

  G_opt.step()

  losses.append(full_batch_loss)
  ## RSR ENDS HERE

  # what follows is for logging/saving/debugging
  if 1:
    print('DEBUG: large_batch_size', large_batch_size, 'epoch', epoch, 'loss', losses[-1],  'lr', scheduler.get_lr())

  if epoch % 20 ==0 and epoch >0:
    if epoch % 100  == 0:
       # save model
       torch.save(G.state_dict(), '%s/G_%06i.pth' % (outdir_weights, epoch))

    count_imgs = 0
    G.eval()
    with torch.no_grad():
     Nb = 11000//batch_size
     for i in range(Nb):
        if i % 100 == 0:
            print('processing batch %i of %i' % (i, Nb))
        z = torch.randn(batch_size, latent_size).cuda()
        fake_images, _ = G([z], randomize_noise=False)
  
        for i in range(batch_size):
           # imageio.imwrite('%s/img_%06i.png' % (outdir_images, count_imgs), fake_images_np[i])
           utils.save_image(
               fake_images[i],
               '%s/img_%06i.png' % (outdir_images, count_imgs),
               nrow=1,
               normalize=True,
               range=(-1, 1),
           )
           count_imgs+=1

    
    G.train()
    print('wrote images to %s' % outdir_images)

    torch.cuda.empty_cache()

    ###################
    # Compute FID score
    # requires https://github.com/mseitzer/pytorch-fid
    fid_command = 'python ../pytorch-fid/pytorch_fid/fid_score.py %s fid_stats_lsun_church.npz  --device cuda:0 ' % outdir_images
    os.system(fid_command)

    # END RSR

