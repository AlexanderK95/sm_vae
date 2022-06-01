import random
import numpy as np
import matplotlib.pyplot as plt

from variational_autoencoder import VAE
from train import load_mnist
from train import load_faces
from train import load_selfmotion
import os
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    plt.show()

def plot_predicted_images(images):
    fig = plt.figure(figsize=(5, 5))
    num_images = len(images)
    for i, image in enumerate(images):
        image = image.squeeze()
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray")
    plt.show()

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


if __name__ == "__main__":
    autoencoder = VAE.load("vae_sm")
    autoencoder.summary()
    # autoencoder.save("ae")
    # x_train, y_train, x_test, y_test = load_mnist()
    x_train, y_train, x_test, y_test = load_selfmotion(5)
    #
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)
    #
    # # num_images = 6000
    # # sample_images, sample_labels = select_images(x_test, y_test, num_images)
    # # _, latent_representations = autoencoder.reconstruct(sample_images)
    # # plot_images_encoded_in_latent_space(latent_representations, sample_labels)
    #
    num_samples = 25
    latent_points = np.array([[random.uniform(-4, 4) for j in range(200)] for i in range(num_samples)])
    new_images = autoencoder.decoder.predict(latent_points)
    plot_predicted_images(new_images)
