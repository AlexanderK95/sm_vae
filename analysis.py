import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skvideo.io
from dataloader import load_selfmotion_vids
from sm_vae import VAE
from sm_vae_c2 import VAE as VAE_c2
import os
from sklearn.manifold import TSNE
import argparse


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.compat.v1.disable_eager_execution()


def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray")
    # plt.show()
    plt.savefig("reconstruction.png")

def plot_predicted_images(images):
    fig = plt.figure(figsize=(5, 5))
    num_images = len(images)
    for i, image in enumerate(images):
        image = image.squeeze()
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray")
    # plt.show()
    plt.savefig("predigion.png")

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

def select_videos(videos, num_vid=10):
    sample_vids_index = np.random.choice(range(len(videos)), num_vid)
    sample_videos = videos[sample_vids_index]
    return sample_videos

def plot_reconstructed_videos(videos, reconstructed_videos):
    pass

def save_video(fname, video):
    skvideo.io.vwrite(fname, video)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--model', help='folder containing parameters and weights of the model')
    # parser.add_argument('--out', help='name for output files')
    parser.add_argument('--grayscale', help='Boolean whether dataset should be loades as grayscale')

    args = parser.parse_args()

    bw = args.grayscale == "True"
    
    output_name = os.path.basename(args.model)
    print(output_name)
    if output_name == "retired":
        sys.exit()

    autoencoder = VAE.load(args.model)
    # autoencoder_c2 = VAE_c2.load("vae_sm_vid_c2")
    # autoencoder.summary()
    # autoencoder_c2.summary()

    video_dim = [8, 512, 512]
    batch_size = 16
    x_train, y_train, x_test, y_test = load_selfmotion_vids("/mnt/masc_home/kressal/datasets/selfmotion/20220930-134704_1.csv", video_dim, batch_size, bw=bw)

    print("###################################################")
    print(args.model)
    print("###################################################")
    print(output_name)
    
    print("Reconstruction")
    num_sample_videos_to_show = 3
    sample_videos = select_videos(x_test, num_sample_videos_to_show)
    reconstructed_videos, latent_points, headings = autoencoder.reconstruct(sample_videos)
    # reconstructed_videos_c2, latent_points_c2 = autoencoder_c2.reconstruct(sample_videos)

    print(reconstructed_videos.shape)
    print(f"Reconstructed:  min: {np.min(reconstructed_videos[0])}      max: {np.max(reconstructed_videos[0])}      mean: {np.mean(reconstructed_videos[0])}")
    print(f"Latent Space:   min: {np.min(latent_points[0])}             max: {np.max(latent_points[0])}             mean: {np.mean(latent_points[0])}")
    
    # print(reconstructed_videos_c2.shape)
    # print(f"Reconstructed:  min: {np.min(reconstructed_videos_c2[0])}      max: {np.max(reconstructed_videos_c2[0])}      mean: {np.mean(reconstructed_videos_c2[0])}")
    # print(f"Latent Space:   min: {np.min(latent_points_c2[0])}             max: {np.max(latent_points_c2[0])}             mean: {np.mean(latent_points_c2[0])}")
    print(reconstructed_videos[0].max())
    print(reconstructed_videos[0].min())
    
    save_video(f"results/{output_name}_recon.mp4", reconstructed_videos[0]*255)
    save_video(f"results/{output_name}_original.mp4", sample_videos[0]*255)
    # save_video('test_c2.mp4', reconstructed_videos_c2[0]*255)


    n_to_show = 5000
    grid_size = 15
    figsize = 12

    mean = np.mean(latent_points, axis=None)
    std = np.std(latent_points, axis=None)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(latent_points.astype("float32"))
    min_x = min(X_tsne[:, 0])
    max_x = max(X_tsne[:, 0])
    min_y = min(X_tsne[:, 1])
    max_y = max(X_tsne[:, 1])


    plt.figure(figsize=(figsize, figsize))
    plt.scatter(X_tsne[:, 0] , X_tsne[:, 1], alpha=0.5, s=2)
    plt.xlabel("Dimension-1", size=20)
    plt.ylabel("Dimension-2", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f"{output_name}\n(mean: {mean}, std: {std}", size=10)
    plt.savefig(f"results/{output_name}_latent_representation.png")

    plot_reconstructed_videos(sample_videos, reconstructed_videos)

    #
    # # num_images = 6000
    # # sample_images, sample_labels = select_images(x_test, y_test, num_images)
    # # _, latent_representations = autoencoder.reconstruct(sample_images)
    # # plot_images_encoded_in_latent_space(latent_representations, sample_labels)
    # #

    print("Prediction")
    num_samples = 10
    latent_points = np.array([[random.uniform(-4, 4) for j in range(420)] for i in range(num_samples)])
    # latent_points = np.array([random.uniform(-4, 4) for j in range(200)])
    print(latent_points.shape)
    new_videos = autoencoder.decoder.predict(latent_points)
    print(new_videos.shape)
    print(f"Predicted:      min: {np.min(new_videos[0])}        max: {np.max(new_videos[0])}        mean: {np.mean(new_videos[0])}")
    print(f"Latent Space:   min: {np.min(latent_points[0])}     max: {np.max(latent_points[0])}     mean: {np.mean(latent_points[0])}")

    save_video(f"results/{output_name}_pred.mp4", new_videos[0]*255)


