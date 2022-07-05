import imageio
import numpy as np
from PIL import Image
import glob
import random
import skvideo


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


def load_selfmotion_vids(target_size, share=100, bw=False, randomize=True):
    print("loading Dataset...")
    file_list = glob.glob('/home/kressal/datasets/selfmotion_vids/*mp4')
    file_list.sort()
    if randomize:
        random.shuffle(file_list)
    num_images_to_load = round(len(file_list) * share / 100)
    print(f"#samples:  {num_images_to_load}")

    # sess = tf.compat.v1.Session()

    # x_train = np.empty(shape=(num_images_to_load, 8, target_size[0], target_size[1], 3)) if not bw else np.empty(shape=(num_images_to_load, 8, target_size[0], target_size[1]))
    x_train = np.empty(shape=(num_images_to_load, 8, target_size[0], target_size[1], 3)).astype("float32") if not bw else np.empty(shape=(num_images_to_load, 8, target_size[0], target_size[1])).astype("float32")
    for n in range(0, num_images_to_load):
        # print(f"processing sample {n}/{num_images_to_load-1}")
        # reader = imageio.get_reader(file_list[n])
        # vid = np.array([img for img in reader])
        # image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        vid = skvideo.io.vread(file_list[n])[0:8,:,:,:]/255.
        _, height, width, _ = vid.shape
        vid = vid[:, int((height-target_size[0])/2):int((height+target_size[0])/2), int((width-target_size[0])/2):int((width+target_size[0])/2), :]
        if bw:
            vid = np.mean(vid, axis=3)
            # vid = vid.reshape(vid.shape + (1,))
            x_train[n, :, :, :] = vid
        else:
            x_train[n, :, :, :, :] = vid

    x_train = x_train.reshape(x_train.shape + (1,)) if bw else x_train
    # x_train = np.mean(x_train, axis=4)

    return x_train