# training
python rsr-features.py

# draw embeddings for baseline
python plot_embeddings.py --load_path checkpoints/e2gan_cifar.pth
# draw embeddings for RSR model
python plot_embeddings.py --load_path checkpoints/RSR_CIFAR10_pretrained.pth
