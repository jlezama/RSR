# RSR

* Source code for  _"Run-Sort-ReRun: Escaping Batch Size Limitations in Sliced Wasserstein Generative Models, J. Lezama, W. Chen, Q. Qiu, ICML 2021"_ 

* This repository contains PyTorch source code to run the toy example in Figure 1 of the paper and CIFAR-10 results in Tables 1 and 2 and Figure 3. 

* Toy Example (Figure 1):
  - run `rsr_toy.py` 
    - optionally, set `large_batch_size` to 4 to simulate Wasserstein distance on 4 samples (3rd subfigure)

* CIFAR10 Example:
  - Based on github repos: `Yuantian013/E2GAN` (included), `mseitzer/pytorch-fid` (included)
  - Download pretrained E2GAN baseline from [this link](https://drive.google.com/drive/folders/1MGJjqsvJBxqfDLlelUarYZUfWTUOwEVt?usp=sharing)
  - Execute `run_training.sh`  or `python rsr-features.py` for training (inside `cifar10`)
  - E2GAN model fine-tuned with RunSort-ReRun available at [this link](https://www.dropbox.com/s/9vluprfcxuhygpi/RSR_CIFAR10_pretrained.pth?dl=0)

* StyleGAN2 Example:
  - Based on github repos: `rosinality/stylegan2-pytorch` (included)
  - Download converted pretrained StyleGAN2 baseline from [this link](https://drive.google.com/file/d/1Nj9yLdxRkvf1U67daWjXsklaEEk9b_f9/view?usp=sharing). (It can also be obtained following the instruction of `rosinality/stylegan2-pytorch`)
  - Download LSUN church using [this link](https://github.com/fyu/lsun), and place it at `stylegan/.data/church/`  
  - Download the stats of LSUN church from [this link](https://drive.google.com/file/d/1IUz_Pzh-7eJlrCvuuKc80kd0LAt_CSne/view?usp=sharing)
  - Execute `run.sh`  or `python rsr-features.py` for training (inside `stylegan`)
  - StyleGAN2 model fine-tuned with RunSort-ReRun available at [this link](https://drive.google.com/file/d/1tsRb57h5kFbeUDvygev596D2eAWmr8IE/view?usp=sharing), place it at `stylegan/weights/`

