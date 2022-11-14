from sm_vae import VAE
from dataloader import SelfmotionDataGenerator
import argparse
import pandas as pd
from create_db import CustomDataGen
import os
import tensorflow as tf
import datetime
import tensorflow.keras as keras



if __name__ == "__main__":
    # logdir = f"logs/fit/debug#


    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--batch-size', help='batch size', default=16)
    parser.add_argument('--epochs', help='number of epochs', default=500)
    parser.add_argument('--kl-weight', help='weight of the KL loss term')
    # parser.add_argument('--dataset-size', help='percentage of dataset to be loaded')
    parser.add_argument('--grayscale', help='Boolean whether dataset should be loades as grayscale')
    parser.add_argument('--recon-loss', help='loss function for the reconstruction loss')
    parser.add_argument('--heading-weight', help='reconstruction loss weight')
    parser.add_argument('--latent-dim', help='dimensionality of latent space')
    parser.add_argument('--learning-rate', help='dimensionality of latent space', default=0.0001)
    parser.add_argument('--res', help='resolution of frames (res x res)', default=256)
    parser.add_argument('--suffix', help='optional, addition to filenames', default="")

    args = parser.parse_args()

    print(args)

    # img_height, img_width = int(args.res), int(args.res)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    # dataset_size = float(args.dataset_size)
    bw = args.grayscale == "True"
    recon_loss = args.recon_loss
    heading_weight = float(args.heading_weight)
    kl_weight = float(args.kl_weight)
    latent_dim = int(args.latent_dim)

    learning_rate = float(args.learning_rate)

    video_dim = [8, int(args.res), int(args.res)]

    suffix = args.suffix if not args.suffix==None else ""

    train_data = SelfmotionDataGenerator("/mnt/masc_home/kressal/datasets/selfmotion/20220930-134704_1_ws.csv", batch_size, video_dim, grayscale=True, shuffle=True)
    val_data = SelfmotionDataGenerator("/mnt/masc_home/kressal/datasets/selfmotion/20220930-134704_1_ws.csv", batch_size, video_dim, grayscale=True, shuffle=True)

    # path = "/home/kressal/datasets/selfmotion_vids"
    # files = [os.path.join(path,fn) for fn in os.listdir(path)]
    # df = pd.DataFrame(files, columns=["filepath"])
    # train_data = CustomDataGen(df, batch_size, input_size=(img_height, img_width))

    # print(train_data.input_shape())

    # vae = VAE(
    #     input_shape=(x_train.shape[1:]),
    #     conv_filters=(64, 64, 64, 32, 16),
    #     conv_kernels=(4, 2, 3, 3, 4),
    #     conv_strides=(2, 2, 2, 1, 1),
    #     latent_space_dim=420
    # )

    grayscale = "bw" if bw else "color"

    name = f"bs_{batch_size}#ep_{epochs}#gs_{bw}#rl_{recon_loss}#hw_{heading_weight}#klw_{kl_weight}#ld_{latent_dim}#lr_{learning_rate}#res_{video_dim[1]}#{suffix}"

    vae = VAE(
        input_shape=(train_data.get_input_shape()),
        conv_filters=(64, 64, 64, 32, 16),
        conv_kernels=([2,5,5], [2,4,4], [2,3,3], [2,3,3], [2,3,3]),
        conv_strides=([1,2,2], [1,2,2], [2,2,2], [2,2,2], [2,1,1]),
        latent_space_dim=latent_dim,
        name=name,
        kl_weight = kl_weight
    )

    # vae = VAE(
    #     input_shape=train_data.input_shape(),
    #     conv_filters=(64, 64, 64, 32, 16),
    #     conv_kernels=([2,27,45], [2,2,2], [2,3,3], [2,3,3], [2,4,4]),
    #     conv_strides=([1,9,15], [1,2,2], [2,2,2], [1,2,2], [1,1,1]),
    #     latent_space_dim=420
    # )

    vae.summary()
    # keras.utils.plot_model(vae.model, to_file="VAEdh.png", show_shapes=True)
    # keras.utils.plot_model(vae.encoder, to_file="Encoder.png", show_shapes=True)
    # keras.utils.plot_model(vae.decoder, to_file="Decoder.png", show_shapes=True)
    # keras.utils.plot_model(vae.heading_decoder, to_file="Heading Decoder.png", show_shapes=True)
    vae.compile(reconstruction_loss=recon_loss, heading_weight=heading_weight, learning_rate=learning_rate)
    
    # vae.train(x_train, [x_train, y_train], batch_size, num_epochs=epochs, grayscale=bw, checkpoint_interval=100)
    # vae.train2(train_data, num_epochs=epochs, checkpoint_interval=int(epochs/10))

    # print(epochs)

    vae.train(train_data, val_data, num_epochs=epochs, grayscale=bw, checkpoint_interval=1500)

    vae.save(f"models/{name}")
    print(f"saved as models/{name}")