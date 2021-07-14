# Adapt from Pytorch Implementation of StyleGAN v2
# https://github.com/rosinality/stylegan2-pytorch

# Download datasets using https://github.com/fyu/lsun
# Place LSUN church at '../.data/church'
# Modify configuration in cfg.py

# Statistical info of LSUN church is fid_stats_lsun_church.npz
# Checkpoint is stylegan2-church-config-f.pt (convert from official ckp)

# Train the experiments
 python rsr-features.py

# Eval
# Our checkpoint is at weights/G_001100.pth
python cal_fid.py