import imageio
import numpy as np
import pandas as pd
from PIL import Image
import glob
import random
from tqdm import tqdm
import skvideo.io as sk
import os
from sm_vae import VAE
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import time

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def load_selfmotion(share=100):
    print("loading Dataset...")
    file_list = glob.glob('N:\\Datasets\selfmotion_imgs\*jpg')
    random.shuffle(file_list)
    num_images_to_load = round(len(file_list) * share / 100)
    print(f"#samples:  {num_images_to_load}")
    x_train = np.array([np.array(Image.open(fname).resize((256,256))) for fname in file_list[0:num_images_to_load-1]])
    x_train = x_train.astype("float32") / 255
    x_train = np.mean(x_train, axis=3)
    x_train = x_train.reshape(x_train.shape + (1,))

    y_train, x_test, y_test = (np.array(range(len(x_train))), x_train, np.array(range(len(x_train))))

    return x_train, y_train, x_test, y_test


def load_selfmotion_vids(path, video_dim, batch_size=32, train_split=0.8, bw=True, randomize=True):
    print("loading Dataset...")

    base_path = os.path.dirname(path)
    data = pd.read_csv(f"{path}", sep=",")

   

    split_idx = int(data.shape[0] * train_split)

    out = np.empty([data.shape[0], video_dim[0], video_dim[1], video_dim[2], 3], dtype=np.uint8) if not bw else np.empty([data.shape[0], video_dim[0], video_dim[1], video_dim[2]], dtype=np.float32)

    idx = np.arange(data.shape[0])
    if randomize:
        np.random.shuffle(idx)

    for i in tqdm(np.arange(data.shape[0])):
        out[idx[i]] = sk.vread(os.path.join(base_path, data["file_head"][i])).squeeze().astype("float32")/255 if not bw else sk.vread(os.path.join(base_path, data["file_head"][i]), as_grey=True).squeeze().astype("float32")/255

    y = data.loc[idx,"velX": "roll"].to_numpy()

    x_train = out[:split_idx]
    y_train = y[:split_idx]
    x_test = out[split_idx:]
    y_test = y[split_idx:]

    x_train = x_train.reshape(x_train.shape + (1,)) if bw else x_train
    x_test = x_test.reshape(x_test.shape + (1,)) if bw else x_test

    return x_train, y_train, x_test, y_test


class SelfmotionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, input_shape=[8, 512, 512], rescale=False, grayscale=True, shuffle=True):
        self.base_path = os.path.dirname(path)
        self.data = pd.read_csv(f"{path}", sep=",")
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.grayscale = grayscale
        self.shuffle = shuffle
        self.rescale = rescale

        self.n = self.data.shape[0]

    def __get_input(self, batch):
        out = np.empty([batch.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2], 3], dtype=np.float32) if not self.grayscale \
              else np.empty([batch.shape[0], self.input_shape[0], self.input_shape[1], self.input_shape[2], 1], dtype=np.float32)

        for i in np.arange(batch.shape[0]):
            vid = sk.vread(os.path.join(self.base_path, batch["file_head"][i])).squeeze().astype("float32")/255. if not self.grayscale \
                     else sk.vread(os.path.join(self.base_path, batch["file_head"][i]), as_grey=True).astype("float32")/255.
            if vid.shape[:-1] != tuple(self.input_shape):
                # print("Input shape does not match video dimensions. Rescaling input to match input shape!")
                for n in np.arange(vid.shape[0]):
                    out[i,n] = tf.image.resize(vid[n], [self.input_shape[1], self.input_shape[2]], antialias=True)
                continue
            out[i] = vid
        return out

    def __get_output(self, batch):
        return batch.loc[:,"velX": "roll"].to_numpy() / np.array([1, 1, 1, 180, 180, 180])

    def __get_data(self, indices):
        batch = self.data.iloc[indices].reset_index(drop=True)
        X = self.__get_input(batch)
        y = self.__get_output(batch)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
    
    def get_input_shape(self):
        input_shape = self.input_shape
        if self.grayscale:
            input_shape.append(1)
        return input_shape

    def __getitem__(self, index):
        indices = np.arange(index * self.batch_size, (index + 1) * self.batch_size)
        X, y = self.__get_data(indices)        
        return X, [X, y]
    
    def __len__(self):
        return self.n // self.batch_size


# if __name__ == "__main__":
#     img_height, img_width = 512, 512
#     batch_size = 16
#     epochs = 250
#     video_dim = [8, 256, 256]
#     bw = True
#     rl = "mse"
#     rlw = 10

#     train_gen = SelfmotionDataGenerator("N:\\Datasets\\selfmotion\\20220930-134704_1.csv", batch_size, video_dim, grayscale=True, shuffle=True)
#     val_gen = SelfmotionDataGenerator("N:\\Datasets\\selfmotion\\20220930-134704_1.csv", batch_size, video_dim, grayscale=True, shuffle=True)
#     test = train_gen.__getitem__(0)

#     # n = 100

#     # start = time.time()
#     # start_p = time.process_time()
#     # for i in tqdm(np.arange(n)):
#     #     SelfmotionDataGenerator("N:/Datasets/selfmotion/20220930-134704_1.csv", batch_size, video_dim, grayscale=True, shuffle=True)
#     # end = time.time()
#     # end_p = time.process_time()
#     # result1 = end-start
#     # result1_p = end_p-start_p

#     # start = time.time()
#     # start_p = time.process_time()
#     # for i in tqdm(np.arange(n)):
#     #     train_gen.__getitem__(0)
#     # end = time.time()
#     # end_p = time.process_time()
#     # result2 = end-start
#     # result2_p = end_p-start_p
#     # print(f"Execution time (Datagen) is {result1 / n} seconds ({result1_p / n} seconds CPU time)")
#     # print(f"Execution time (fetching batch) is {result2 / n} seconds ({result2_p / n} seconds CPU time)")

#     # x_train, y_train, x_test, y_test = load_selfmotion_vids("N:\\Datasets\\selfmotion\\20220930-134704_1.csv", video_dim, batch_size, randomize=False)
    
#     # image1 = test[0][0,0,:]
#     # image2 = x_train[0,0,:]

#     # plt.figure()
#     # plt.imshow(image1)
#     # plt.figure()
#     # plt.imshow(image2)
#     # plt.show()


#     vae = VAE(
#         input_shape=(train_gen.get_input_shape()),
#         conv_filters=(64, 32, 16),
#         conv_kernels=([2,6,6], [2,4,3], [2,4,3]),
#         conv_strides=([2,4,4], [2,2,2], [1,2,2]),
#         latent_space_dim=180,
#         name="test"
#     )

#     # vae = VAE(
#     #     input_shape=(x_train.shape[1:]),
#     #     conv_filters=(64, 64, 64, 32, 16),
#     #     conv_kernels=([2,5,5], [2,4,4], [2,3,3], [2,3,3], [2,3,3]),
#     #     conv_strides=([1,2,2], [1,2,2], [2,2,2], [2,2,2], [2,1,1]),
#     #     latent_space_dim=420,
#     #     name="test"
#     # )

#     vae.summary()
#     keras.utils.plot_model(vae.model, to_file="VAEdh.png", show_shapes=True)
#     keras.utils.plot_model(vae.encoder, to_file="Encoder.png", show_shapes=True)
#     keras.utils.plot_model(vae.decoder, to_file="Decoder.png", show_shapes=True)
#     keras.utils.plot_model(vae.heading_decoder, to_file="Heading Decoder.png", show_shapes=True)
#     # keras.utils.plot_model(vae.embedding_stats, to_file="self.embedding_stats.png", show_shapes=True)
#     vae.compile(reconstruction_loss=rl, reconstruction_weight=rlw)

#     vae.train(train_gen, val_gen, batch_size, num_epochs=epochs, grayscale=bw, checkpoint_interval=100)
#     vae.save(f"models/batch-size_{batch_size}#epochs_{epochs}#grayscale_{bw}#recon-loss_{rl}#recon-weight_{rlw}#test")
#     pass