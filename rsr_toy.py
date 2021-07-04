# -*- coding: utf-8 -*-
"""
Source code of ICML 2021 paper "Run-Sort-ReRun: Escaping Batch Size Limitations in Sliced Wasserstein Generative Models"

Implementation of 1-D Run-Sort-ReRun for the toy example in Figure 1.

Copyright 2021 jlezama@fing.edu.uy
"""


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import datetime
import os, sys


import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_real(batch_size, K=4):
  """
  function to create 1D "ground truth" multi-modal dataset
  K is the number of modes
  """

  mus = np.arange(K)
  x = [];
  for _ in range(batch_size):
    i = np.random.randint(len(mus))
    x.append (np.random.randn(1,1)*.2 + np.asarray([i]))

  return np.asarray(x).reshape(batch_size,1)-np.asarray([(K-1)/2.,])




class RunningBN(nn.Module):
    """BatchNorm layer that tracks mean and var with very small batch size"""
    
    def __init__(self, size_in, batch_size=1024, alpha=0.9):
        super().__init__()
        self.size_in = size_in
        
        self.running_mean = torch.zeros(size_in).cuda()
        self.running_var = torch.ones(size_in).cuda()


        self.running_mean_aux = torch.zeros(size_in).cuda()
        self.running_var_aux = torch.zeros(size_in).cuda()

        self.alpha =  alpha 
        self.eps = 1e-5
        self.batch_size = batch_size
        self.count = 0

        scale = torch.ones(size_in).cuda()
        self.scale = nn.Parameter(scale)  
        shift = torch.zeros(size_in).cuda()
        self.shift = nn.Parameter(shift)

    def forward(self, x, keepstats=True):

        x_norm = (x - self.running_mean)/torch.sqrt(self.running_var + self.eps) 

        if keepstats:
          # accumulate stats while visiting small batches
          sum_ = torch.sum(x.detach(), dim=0)
          q = x.detach() - self.running_mean                       
          qsq = q*q                            
          var = torch.sum(qsq, dim=0)
          self.running_mean_aux += sum_  
          self.running_var_aux += var
          self.count += x.shape[0]        

        return x_norm  *self.scale + self.shift

    def update(self):
      # update running mean and var
      alpha = self.alpha
      self.running_mean = alpha*self.running_mean + (1-alpha)*self.running_mean_aux.clone() / float(self.count)
      self.running_var = alpha*self.running_var + (1-alpha) * self.running_var_aux.clone() / float(self.count)
      self.running_mean_aux = torch.zeros(self.size_in).cuda()
      self.running_var_aux = torch.zeros(self.size_in).cuda()
      self.count = 0

    def reset_count(self):
      self.running_mean_aux = torch.zeros(self.size_in).cuda()
      self.running_var_aux = torch.zeros(self.size_in).cuda()
      self.count = 0


class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, input_size=1, num_classes=1, alpha=0.9):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = RunningBN(128, alpha=alpha)  
        self.fc5 = nn.Linear(128, self.num_classes)
        
        
    def forward(self, x, keepstats=True):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.bn4(x, keepstats)
        x = self.fc5(x)

        y_ = x.view(x.size(0), self.num_classes)
        return y_






""" main algorithm """
max_epoch = 20000 
alpha = 0.9 # batchnorm MA
n_noise = 1
norm = 'L1'
large_batch_size = 512
small_batch_size = 4 
        

G = Generator(n_noise, alpha=alpha).to(DEVICE)
G.train()
        
G_opt = torch.optim.Adam(G.parameters(), lr=1e-4, weight_decay=0)
        
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_opt, max_epoch, eta_min=1e-7)
        
K = 4 # number of modes in 1-D dataset
        
        
print_every = 1000
        
losses = []
        



for epoch in range(max_epoch):
    scheduler.step()

    G.bn4.reset_count()

    # STEP 1 RUN
    all_z = None
    all_output = None

    for idx in range(int(large_batch_size/small_batch_size)):
        # Training generator

        with torch.no_grad():
          z = torch.randn(small_batch_size, n_noise).to(DEVICE) # sample noise
          output = G(z) # run through model
        
        if all_output is None:
          all_z = z.detach()
          all_output = output.detach()
        else:
          all_z = torch.cat((all_z, z.detach()), dim=0)
          all_output = torch.cat((all_output, output.detach()), dim=0)

    all_output = all_output.view(large_batch_size, -1)
    all_z = all_z.view(large_batch_size, -1)
    all_gt = torch.Tensor(sample_real(large_batch_size, K=K)).to(DEVICE)

    # STEP 2 SORT
    [gt_sort_val, gt_sort_ix] = torch.sort(all_gt, dim=0)

    [out_sort_val, out_sort_ix] = torch.sort(all_output, dim=0)
    all_z = all_z[out_sort_ix].view(large_batch_size, -1)


    # STEP 3 RERUN
    G_opt.zero_grad()

    full_batch_loss = 0
    all_output_new = None
    for idx in range(0,large_batch_size, small_batch_size):
      gt = gt_sort_val[idx:idx+small_batch_size]  
      z = all_z[idx:idx+small_batch_size]
      output = G(z)

      if norm == 'L1':
        G_loss = torch.mean(torch.abs(gt-output)) # L1
      else:
        G_loss = torch.mean((gt-output)**2) # L2
        
      G_loss.backward()

      full_batch_loss += G_loss.item()/(large_batch_size/small_batch_size)

    G_opt.step()

    G.bn4.update() # update BN running mean and var
 
    losses.append(full_batch_loss)

    if epoch % print_every==0:
           print('iter %i/%i,  G_loss_queue %2.4e, lr %2.2e'  %  (epoch, max_epoch,  losses[-1], scheduler.get_lr()[0] ) )

           
plt.figure()
plt.plot(np.log10(np.asarray(losses)))
plt.grid()
_ = plt.title('Training loss')

plt.savefig( 'loss.pdf' )



fig, axs = plt.subplots(2, figsize = (5,10))
fig.suptitle('True vs Generated Histogram')

all_gt = sample_real(10240, K=K)
z = torch.randn(10240, n_noise).to(DEVICE)
output = G(z) 

real = all_gt.squeeze()
fake = output.detach().cpu().numpy().squeeze()

xlim = 6
_ = axs[0].hist(real, bins=np.linspace(-xlim,xlim,num=100))
_  = axs[1].hist(fake, bins=np.linspace(-xlim,xlim,num=100))



plt.savefig('hists.pdf')

plt.close('all')

        
