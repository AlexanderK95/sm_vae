from tensorflow.keras.datasets import mnist

from variational_autoencoder import VAE
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import random

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
# mirrored_strategy = tf.distribute.experimental.CentralStorageStrategy(devices=["/gpu:0","/gpu:1"])

LEARNING_RATE = 0.00001
BATCH_SIZE = 64
EPOCHS = 1000

gpus=tf.config.get_visible_devices('GPU')
tf.config.set_visible_devices(gpus[1],'GPU')


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test

def load_faces():
    file_list = glob.glob('/home/alexanderk/Documents/Datasets/faces/*jpg')
    x_train = np.array([np.array(Image.open(fname).resize((256,256))) for fname in file_list])
    x_train = x_train.astype("float32") / 255
    # x_train = np.mean(x_train, axis=3)
    # x_train = x_train.reshape(x_train.shape + (1,))

    y_train, x_test, y_test = (np.array(range(len(x_train))), x_train, np.array(range(len(x_train))))

    return x_train, y_train, x_test, y_test

def load_selfmotion(share=100):
    file_list = glob.glob('/home/alexanderk/Documents/Datasets/selfmotion_imgs/selfmotion_imgs/*jpg')
    random.shuffle(file_list)
    num_images_to_load = round(len(file_list) * share / 100)
    x_train = np.array([np.array(Image.open(fname).resize((256,256))) for fname in file_list[0:num_images_to_load-1]])
    x_train = x_train.astype("float32") / 255
    x_train = np.mean(x_train, axis=3)
    x_train = x_train.reshape(x_train.shape + (1,))

    y_train, x_test, y_test = (np.array(range(len(x_train))), x_train, np.array(range(len(x_train))))

    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 256, 3),
        conv_filters=(32, 64, 64, 64, 64),
        conv_kernels=(2, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, 2),
        latent_space_dim=200
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    # x_train, _, _, _ = load_mnist()
    # autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("vae")

    # print("loading model...")
    # autoencoder = VAE.load("model")
    # print("compiling model...")
    # autoencoder.compile(LEARNING_RATE)
    # print("trainig model...")
    # autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
    # print("saving model")
    # autoencoder.save("model")
    with tf.device("/gpu:1"):
        x_train, _, _, _ = load_selfmotion()
        # plt.imshow(x_train[0])
        # plt.show()
        # with mirrored_strategy.scope():
        autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
        # autoencoder = VAE.load("vae_faces")
        # autoencoder.compile(LEARNING_RATE)
        # autoencoder.train(x_train, BATCH_SIZE, EPOCHS)
        autoencoder.save("vae_sm2")
