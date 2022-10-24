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

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def load_selfmotion(share=100):
    print("loading Dataset...")
    file_list = glob.glob('E:\\Datasets\selfmotion_imgs\dump\*jpg')
    random.shuffle(file_list)
    num_images_to_load = round(len(file_list) * share / 100)
    print(f"#samples:  {num_images_to_load}")
    x_train = np.array([np.array(Image.open(fname).resize((256,256))) for fname in file_list[0:num_images_to_load-1]])
    x_train = x_train.astype("float32") / 255
    x_train = np.mean(x_train, axis=3)
    x_train = x_train.reshape(x_train.shape + (1,))

    y_train, x_test, y_test = (np.array(range(len(x_train))), x_train, np.array(range(len(x_train))))

    return x_train, y_train, x_test, y_test


def load_selfmotion_vids(path, video_dim, batch_size=32, train_split=0.8, bw=False, randomize=True):
    print("loading Dataset...")

    base_path = os.path.dirname(path)
    data = pd.read_csv(f"{path}", sep=",")

   

    split_idx = int(data.shape[0] * train_split)

    out = np.empty([data.shape[0], video_dim[0], video_dim[1], video_dim[2], 3], dtype=np.uint8) if not bw else np.empty([data.shape[0], video_dim[0], video_dim[1], video_dim[2]], dtype=np.float32)

    idx = np.arange(data.shape[0])
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



# if __name__ == "__main__":
#     img_height, img_width = 512, 512
#     batch_size = 16
#     epochs = 50
#     video_dim = [8, 512, 512]
#     bw = True
#     rl = "mse"
#     rlw = 10

#     x_train, y_train, x_test, y_test = load_selfmotion_vids("E:\\Datasets\\selfmotion\\20220930-134704_1.csv", video_dim, batch_size)

#     x_train2, y_train2, x_test2, y_test2 = load_selfmotion(1)

#     vae = VAE(
#         input_shape=(x_train.shape[1:]),
#         conv_filters=(64, 64, 64, 32, 16),
#         conv_kernels=([2,5,5], [2,4,4], [2,3,3], [2,3,3], [2,3,3]),
#         conv_strides=([1,2,2], [1,2,2], [2,2,2], [2,2,2], [2,1,1]),
#         latent_space_dim=420,
#         name="test"
#     )

#     vae.summary()
#     vae.compile(reconstruction_loss=rl, reconstruction_weight=rlw)

#     vae.train(x_train, [x_train, y_train], batch_size, num_epochs=epochs, grayscale=bw, checkpoint_interval=100)

#     pass