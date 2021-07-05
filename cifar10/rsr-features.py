# -*- coding: utf-8 -*-
"""
Source code for training Run-Sort-ReRun on CIFAR-10 

Adapted from   https://github.com/Yuantian013/E2GAN

Copyright 2021 jlezama@fing.edu.uy

"""
import imageio
import cfg
import models_search
from inception import InceptionV3
import numpy as np
import os
import torch
from torchvision import transforms
import torchvision.datasets as dset


def main():
    args = cfg.parse_args()

    # load Inception feature extractor
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.d_feature]
    inception_model = InceptionV3([block_idx], normalize_input=False)
    inception_model.cuda()
    inception_model.eval()

    # download CIFAR-10 statists
    if not os.path.isfile('fid_stats_cifar10_train.npz'):
        os.system('wget http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz')

    # load the E2GAN models
    args.arch = [0, 1, 0, 1, 0, 1, 2, 1, 0, 0, 1, 0, 1, 2]  # e2gan architecture is defined this way (see paper)
    G = eval('models_search.' + args.gen_model + '.Generator')(args=args).cuda()
    G.set_arch(args.arch, cur_stage=2)

    # load weights

    args.load_path = 'checkpoints/e2gan_cifar.pth' # Download pretrained E2GAN baseline from https://drive.google.com/drive/folders/1MGJjqsvJBxqfDLlelUarYZUfWTUOwEVt?usp=sharing

    # args.load_path = 'checkpoints/RSR_CIFAR10_pretrained.pth' # optionally download pre-trained model from https://www.dropbox.com/s/9vluprfcxuhygpi/RSR_CIFAR10_pretrained.pth?dl=0

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

    print(G)
    G.train()
    G = G.cuda()

    # optimizer
    max_epoch = args.max_epoch
    G_opt = torch.optim.Adam(G.parameters(), lr=args.g_lr, weight_decay=0)  # set LR and weight decay here
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_opt, max_epoch)

    description = 'full_lbs_%i_%i' % (args.lbs_init, args.lbs_init)
    print('running experiment %s' % description)
    outdir_images = 'outimgs/fakes/%s/' % (description)
    outdir_weights = 'weights/%s/' % (description)
    os.system('mkdir -p %s' % outdir_images)
    os.system('mkdir -p %s' % outdir_weights)

    batch_size = args.gen_batch_size
    latent_size = args.latent_dim
    small_batch_size = batch_size
    large_batch_size = args.lbs_init

    NS = large_batch_size // small_batch_size

    N_rotmat = args.n_rotmat  # number of random projections
    d_img = args.d_feature  # inception feature dimension

    # dataset
    dataset = dset.CIFAR10(root='.data/', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=small_batch_size,
                                             shuffle=True, num_workers=2, drop_last=True)
    dataloader_iterator = iter(dataloader)

    losses = []

    for epoch in range(max_epoch):

        scheduler.step()

        # get projection matrix
        if epoch == 0:
            rotmat_img = torch.randn(d_img, N_rotmat).cuda()
            rotmat_img = rotmat_img / torch.sqrt(torch.sum(rotmat_img ** 2, dim=0))
        else:
            with torch.no_grad():
                pworst = 1 / 3.0
                # keep 1/3rd of worst projections, add 2/3rd new ones

                worst_values_img, worst_index_img = torch.sort(G_loss_all_img, descending=True)
                rotmat_img_prev = rotmat_img[:, worst_index_img[:int(N_rotmat * pworst)]]

                # rotmatimg will be taken from pairs of gt, output
                N_rotmat_new = int(N_rotmat * (1 - pworst))
                ix_gt = np.random.randint(0, large_batch_size, N_rotmat_new)
                ix_output = np.random.randint(0, large_batch_size, N_rotmat_new)

                vectors_gt = all_gt[ix_gt, :].detach().t().cuda()
                vectors_out = all_output_img[ix_output, :].detach().t()

                rotmat_img = (vectors_gt - vectors_out)

                # normalize
                rotmat_img = rotmat_img / torch.sqrt(torch.sum(rotmat_img ** 2, dim=0))
                rotmat_img = torch.cat((rotmat_img, rotmat_img_prev), dim=1)

        # initialize tensors for noise vectors, real data and fake data
        all_z = torch.randn(large_batch_size, latent_size).cuda()
        all_gt = torch.zeros(large_batch_size, d_img).cuda()
        all_output_img = torch.zeros(large_batch_size, d_img).cuda()

        ####################################################
        # STEP 1. RUN
        with torch.no_grad():
            for idx in range(NS):
                try:
                    images, _ = next(dataloader_iterator)
                except:
                    dataloader_iterator = iter(dataloader)
                    images, _ = next(dataloader_iterator)
                images = images.cuda()
                inception_features_gt = inception_model(images)[0].view(batch_size, -1)
                all_gt[idx * batch_size:(idx + 1) * batch_size, :] = inception_features_gt

                z = all_z[idx * batch_size:(idx + 1) * batch_size, :]
                fake_images = G(z)

                # compute inception feature
                inception_features = inception_model(fake_images)[0].view(batch_size, -1)
                all_output_img[idx * batch_size:(idx + 1) * batch_size, :] = inception_features

        # finished computing features, now project
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
        G_loss_all_img = 0

        # now do actual comparison
        for idx in range(0, large_batch_size, small_batch_size):
            z = all_z[idx:idx + small_batch_size, :]
            fake_images = G(z)

            # compute inception feature
            inception_features = inception_model(fake_images)[0].view(batch_size, -1)
            output_img = inception_features.mm(rotmat_img)  # project

            # get the relative position of the output
            rel_ix_img = out_img_sort_relative[idx:idx + small_batch_size, :]

            # now get the equivalent positions of the gt
            gt = gt_sort_val.gather(0, rel_ix_img).cuda()

            diff_img = (gt - output_img) ** 2

            threshold_img = 0  # Optional: don't penalize too small differences, this is normal even for samples of the same distribution.
            
            diff_img = (torch.clamp(diff_img, min=threshold_img) - threshold_img)

            G_loss_row_img = torch.sum(diff_img, dim=0) / large_batch_size
            G_loss_img = torch.sum(G_loss_row_img) / rotmat_img.shape[1]

            G_loss = G_loss_img
            G_loss.backward()

            G_loss_all_img += G_loss_row_img.detach().cpu()
            full_batch_loss += G_loss.item()

        G_opt.step()
        losses.append(full_batch_loss)
        ## RSR ENDS HERE

        #######################################################3
        # what follows is for logging/saving/validatingdebugging
        if epoch % 100 == 0:
            print('DEBUG: large_batch_size', large_batch_size, 'epoch', epoch, 'loss', losses[-1], 'lr', scheduler.get_lr())

        # evaluation: calculate FID
        if epoch % 50 == 0 and epoch > 0:
            if epoch % 100 == 0:
                # save model
                torch.save(G.state_dict(), '%s/G_%06i.pth' % (outdir_weights, epoch))

            count_imgs = 0
            G.eval()
            with torch.no_grad():
                Nb = 11000 // batch_size # write 10K sampled images for test FID computation
                for i in range(Nb):
                    print('processing batch %i of %i' % (i, Nb))
                    z = torch.randn(batch_size, latent_size).cuda()
                    fake_images = G(z)

                    fake_images_np = fake_images.cpu().detach().numpy()

                    fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 3, 32, 32)
                    fake_images_np = ((fake_images_np.transpose((0, 2, 3, 1)) / 2.0 + .5) * 255).astype(np.uint8)

                    for i in range(batch_size):
                        imageio.imwrite('%s/img_%06i.png' % (outdir_images, count_imgs), fake_images_np[i])
                        count_imgs += 1

            G.train()
            print('wrote images to %s' % outdir_images)

            torch.cuda.empty_cache()

            ###################
            # Compute FID score
            # requires https://github.com/mseitzer/pytorch-fid
            
            fid_command = 'python pytorch-fid/pytorch_fid/fid_score.py %s fid_stats_cifar10_train.npz  --device cuda:0 ' % outdir_images
            os.system(fid_command)



if __name__ == '__main__':
    main()
