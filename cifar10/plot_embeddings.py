import cfg
import sys, os
import torch
from inception import InceptionV3
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dset
from torchvision import transforms
import copy
import models_search

dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
inception_model = InceptionV3([block_idx], normalize_input=False)
inception_model.cuda()
inception_model.eval()

# load the E2GAN models
args = cfg.parse_args()

args.arch = [0, 1, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2] # e2gan architecture is defined this way (see paper)

G = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
G.set_arch(args.arch, cur_stage=2)
G.eval()

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

dataset = dset.CIFAR10(root='.data/', download=True,
                        transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

batch_size = 20
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2, drop_last=True)

dataloader_iterator = iter(dataloader)

src = []
tgt = []
with torch.no_grad():
    for i, (images, _) in enumerate(dataloader):
        z = torch.randn(batch_size, 128).cuda()
        tgt.append(copy.deepcopy(inception_model(images.cuda())[0].view(batch_size, -1)).cpu().numpy())
        fake_images = G(z)
        src.append(copy.deepcopy(inception_model(fake_images)[0].view(batch_size, -1)).cpu().numpy())

src_np = np.concatenate(tuple(src), axis=0)
tgt_np = np.concatenate(tuple(tgt), axis=0)

if not os.path.exists("figs/"):
    os.mkdir("figs")

all_np = np.concatenate((src_np, tgt_np), axis=0)

pca = PCA(n_components=2)
pc = pca.fit(tgt_np)

src_pca = pca.transform(src_np)
tgt_pca = pca.transform(tgt_np)

plt.figure(figsize=(15, 15))
plt.scatter(tgt_pca[:, 0], tgt_pca[:, 1], 4, 'g', alpha=0.4)
plt.grid()
plt.xlim([-11, 11])
plt.ylim([-11, 11])
plt.savefig('figs/tgt.png')

plt.figure(figsize=(15, 15))
plt.scatter(src_pca[:, 0], src_pca[:, 1], 4, 'r', alpha=0.4)
plt.grid()
plt.xlim([-11, 11])
plt.ylim([-11, 11])
plt.savefig('figs/src.png')

plt.close('all')
