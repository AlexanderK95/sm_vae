# print("inside vae.py")
import os
import time
import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
# from keras import layers
from tensorflow.keras import backend as K
# from IPython import display
import datetime
import math

from PIL import Image
import glob
import random

import pandas as pd

from create_db import CustomDataGen

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(True)


class VAE:
    def __init__(self,input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        print("initializing vae...")
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 10000

        self.dataset = None
        self.encoder = None
        self.decoder = None
        self.model = None

        self._optimizer = tf.keras.optimizers.Adam(lr = 0.0001)
        self._checkpoint_filepath = './tmp/checkpoint'

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    @classmethod
    def load(cls, save_folder="."):
        pass

    def load_dataset(self):
        pass

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        # optimizer = Adam(learning_rate=learning_rate)
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=self._combined_loss,
            metrics=[self._mse_loss,
                    self._kl_loss])


    def train(self, train_ds, val_ds, num_epochs, checkpoint_interval=50):
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="tmp/checkpoints/",
            monitor='loss',
            verbose = 1,
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            save_freq = 'epoch',
            period = checkpoint_interval)
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.model.fit(
            train_ds,
            # validation_data=val_ds,
            # batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True,
            callbacks=[
                self.tensorboard_callback,
                model_checkpoint_callback,
                tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 * math.exp(-0.001*epoch))
            ]
        )

    def train2(self, train_ds, num_epochs, checkpoint_interval=50):
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="tmp/checkpoints/",
            monitor='loss',
            verbose = 1,
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            save_freq = 'epoch',
            period = checkpoint_interval)
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.model.fit(
            train_ds,
            epochs=num_epochs,
            callbacks=[
                self.tensorboard_callback,
                model_checkpoint_callback,
                # tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 * math.exp(-0.001*epoch))
            ]
        )

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = tf.keras.Input(shape=self.input_shape, name='input_layer')
        self._model_input = encoder_input
        x = encoder_input

        # Normalization layer
        # x = tf.keras.layers.Rescaling(1./255, input_shape=self.input_shape, name="norm_layer")(x)

        # Convolution blocks (conv layers + (leaky) ReLU + Batch norm)
        for layer_index in range(self._num_conv_layers):
            layer_number = layer_index + 1
            x = tf.keras.layers.Conv3D(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"conv_{layer_number}"
            )(x)
            # x = layers.ReLU(name=f"relu_{layer_number}")(x)
            x = tf.keras.layers.LeakyReLU(name=f"lrelu_{layer_number}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_{layer_number}")(x)

        # Final Block
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        flatten = tf.keras.layers.Flatten()(x)
        self.mean = tf.keras.layers.Dense(self.latent_space_dim, name='mean')(flatten)
        self.log_var = tf.keras.layers.Dense(self.latent_space_dim, name='log_var')(flatten)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mean), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        output = tf.keras.layers.Lambda(sample_point_from_normal_distribution, name="lambda")([self.mean, self.log_var])
        self.encoder = tf.keras.Model(encoder_input, output, name="Encoder")

    def _build_decoder(self):
        num_neurons = np.prod(self._shape_before_bottleneck)

        decoder_input = tf.keras.Input(shape=self.latent_space_dim, name='input_layer')
        dense_layer = tf.keras.layers.Dense(num_neurons, name="dense_1")(decoder_input)
        x = tf.keras.layers.Reshape(self._shape_before_bottleneck, name='Reshape')(dense_layer)

        for layer_index in reversed(range(1, self._num_conv_layers)):
            layer_num = self._num_conv_layers - layer_index
            x = tf.keras.layers.Conv3DTranspose(
                filters=self.conv_filters[layer_index],
                kernel_size=self.conv_kernels[layer_index],
                strides=self.conv_strides[layer_index],
                padding="same",
                name=f"conv_transpose_{layer_num}"
            )(x)
            # x = layers.ReLU(name=f"relu_{layer_num}")(x)
            x = tf.keras.layers.LeakyReLU(name=f"lrelu_{layer_num}")(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_{layer_num}")(x)

        output = tf.keras.layers.Conv3DTranspose(
            filters=self.input_shape[2],
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            activation='sigmoid',
            name=f"conv_transpose_{self._num_conv_layers}"
        )(x)

        self.decoder = tf.keras.Model(decoder_input, output, name="Decoder")

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = tf.keras.Model(model_input, model_output, name="VAE")

    def _combined_loss(self, y_true, y_pred):
        reconstruction_loss = self._mse_loss(y_true, y_pred)
        kl_loss = self._kl_loss(y_true, y_pred)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        return combined_loss

    def _mse_loss(self, y_true, y_pred):
        r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
        return r_loss

    def _kl_loss(self, y_true, y_pred):
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mean) - K.exp(self.log_var), axis = 1)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)








# def generate_and_save_images(model, epoch, test_input):
#   # Notice `training` is set to False.
#   # This is so all layers run in inference mode (batchnorm).
#     mean, var = enc(test_input, training=False)
#     latent = final([mean, var])
#     predictions = dec(latent, training=False)
#     print(predictions.shape)
#     fig = plt.figure(figsize=(4,4))

#     for i in range(predictions.shape[0]):
#         plt.subplot(5, 5, i+1)
#         pred = predictions[i, :, :, :] * 255
#         pred = np.array(pred)
#         pred = pred.astype(np.uint8)
#         #cv2.imwrite('tf_ae/images/image'+ str(i)+'.png',pred)

#         plt.imshow(pred)
#         plt.axis('off')

#     plt.savefig('tf_vae/scene_imgs/images/image_at_epoch_{:d}.png'.format(epoch))
#     plt.show()

def load_selfmotion(share=100):
    print("loading Dataset...")
    file_list = glob.glob('E:\Datasets\selfmotion_imgs\dump\*jpg')
    random.shuffle(file_list)
    num_images_to_load = round(len(file_list) * share / 100)
    print(f"#samples:  {num_images_to_load}")
    x_train = np.array([np.array(Image.open(fname).resize((256,256))) for fname in file_list[0:num_images_to_load-1]])
    x_train = x_train.astype("float32") / 255
    x_train = np.mean(x_train, axis=3)
    x_train = x_train.reshape(x_train.shape + (1,))

    y_train, x_test, y_test = (np.array(range(len(x_train))), x_train, np.array(range(len(x_train))))

    return x_train, y_train, x_test, y_test



if __name__ == "__main__":
    
    print(tf.__version__)
    print("running main...")

    img_height, img_width = 256, 256
    batch_size = 16


    path = "E:\\Datasets\\selfmotion_vids"
    files = [os.path.join(path,fn) for fn in os.listdir(path)]
    df = pd.DataFrame(files, columns=["filepath"])
    train_data = CustomDataGen(df, batch_size)

    vae = VAE(
        input_shape=(9, img_height, img_width, 3),
        conv_filters=(64, 128, 64, 64, 32),
        conv_kernels=(4, 4, 3, 3, 4),
        conv_strides=(2, 2, 2, 2, 2),
        latent_space_dim=200
    )

    vae.summary()
    vae.compile()
    
    # vae.train(train_ds, val_ds, 10, checkpoint_interval=1)
    vae.train2(train_data, num_epochs=10)
    # vae.save("vae_sm2")
    pass

