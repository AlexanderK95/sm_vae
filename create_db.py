import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

# tf.compat.v1.disable_eager_execution()
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df,
                 batch_size,
                 input_size=(512, 512),
                 img_count=8,
                 grayscale=False,
                 shuffle=True):
        
        self.df = df.copy()
        # self.X_col = X_col
        # self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.img_count = img_count
        self.grayscale = grayscale
        self.shuffle = shuffle
        
        self.n = len(self.df)
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def input_shape(self):
        return (self.img_count, self.input_size[0], self.input_size[1], 3)
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__get_data(batches)
        return X, X
    
    def __len__(self):
        return self.n // self.batch_size


    def __get_input(self, path, target_size):

        reader = imageio.get_reader(path[0])
        cnt = 0
        vid = np.empty(shape=(self.img_count, target_size[0], target_size[1], 3)).astype("float32") if not self.grayscale else np.empty(shape=(self.img_count, 8, target_size[0], target_size[1])).astype("float32")
        for img in reader:
            if cnt < self.img_count:
                vid[cnt,:,:,:] = np.array(img)
                cnt += 1

        # image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        # vid_arr = tf.compat.v1.Session().run(tf.image.resize(vid,(target_size[0], target_size[1])))
        # vid_arr = tf.image.resize(vid,(target_size[0], target_size[1]))
        # print(vid.shape)
        return vid/255.

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches["filepath"]

        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in zip(path_batch)])

        return X_batch

   


# if __name__ == "__main__":

#     path = "E:\\Datasets\\vid_test"
#     files = [os.path.join(path,fn) for fn in os.listdir(path)]
#     df = pd.DataFrame(files, columns=["filepath"])

#     datagen = CustomDataGen(df, batch_size=2)

#     test = datagen[0]

#     pass



    # reader = imageio.get_reader(f"{path}\\eyeMovementSim1.mp4")
    # vid = np.array([img for img in reader])
    # vid.shape
    # vid2 = np.mean(vid, axis=3)

    # vid_arr = tf.image.resize(vid,(512,512)).numpy().astype("float32")/255

    # print(f"3 channels: {vid.size/1000000}MB, 1 channel: {vid2.size/1000000}MB, ratio: {vid.size/vid2.size}")

    # plot = None
    # for i in range(0, vid_arr.shape[0]):
    #     plot = plt.imshow(vid[i,:,:,:])
    #     plot = plt.imshow(vid_arr[i,:,:,:])
    #     plt.pause(0.1)
    #     plt.draw()
    # pass

