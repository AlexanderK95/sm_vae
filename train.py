from sm_vae import VAE
from dataloader import load_selfmotion_vids
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--batch-size', help='batch size')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--dataset-size', help='percentage of dataset to be loaded')
    parser.add_argument('--grayscale', help='Boolean whether dataset should be loades as grayscale')
    parser.add_argument('--recon-loss', help='loss function for the reconstruction loss')
    parser.add_argument('--recon-weight', help='reconstruction loss weight')

    args = parser.parse_args()

    print(args)

    img_height, img_width = 256, 256
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    dataset_size = float(args.dataset_size)
    bw = args.grayscale == "True"
    rl = args.recon_loss
    rlw = float(args.recon_weight)

    x_train = load_selfmotion_vids([img_height, img_width], dataset_size, bw)

    # path = "E:\\Datasets\\selfmotion_vids"
    # files = [os.path.join(path,fn) for fn in os.listdir(path)]
    # df = pd.DataFrame(files, columns=["filepath"])
    # train_data = CustomDataGen(df, batch_size, input_size=(img_height, img_width))

    vae = VAE(
        input_shape=(x_train.shape[1:]),
        conv_filters=(64, 64, 64, 32, 16),
        conv_kernels=(4, 2, 3, 3, 4),
        conv_strides=(2, 2, 2, 1, 1),
        latent_space_dim=420
    )

    vae.summary()
    vae.compile(reconstruction_loss=rl, reconstruction_weight=rlw)
    
    vae.train(x_train, batch_size, num_epochs=epochs, checkpoint_interval=100)
    # vae.train2(train_data, num_epochs=10)

    grayscale = "bw" if bw else "color"
    vae.save(f"vae_sm_vid_{bw}_{rl}")