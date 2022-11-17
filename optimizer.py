import numpy as np
# from sklearn.metrics import mean_squared_error as mse
import random
from tqdm import tqdm
from sm_vae import VAE
from dataloader import SelfmotionDataGenerator
from analysis import save_video
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import argparse


def mse(Y_true, Y_pred):
    return np.square(np.subtract(Y_true,Y_pred)).mean()

def ssmi(Y_true, Y_pred):
    index = 0
    for i in np.arange(Y_true.shape[0]):
        index += structural_similarity(Y_true[i], Y_pred[i])
    index = index / Y_true.shape[0]
    return index


class FDGDOptimizer:
    def __init__(self, sample_size:int=50, generator=None, learning_rate=0.1, search_radius=0.1, ground_truth=None):
        self.n = sample_size
        self.generator = generator
        self.sigma = search_radius
        self.learning_rate = learning_rate

        self.c0 = np.array([random.uniform(-2, 2) for j in range(int(self.generator.latent_space_dim))])

        self.y_true = ground_truth


    def step(self):
        ldim = self.generator.latent_space_dim
        # pertubations = np.zeros([self.n, ldim])
        # pertubations = np.zeros([[random.uniform(-2, 2) for j in range(int(ldim))] for i in range(num_samples)])
        self.delta_y = np.zeros(self.n)
        delta_c0 = np.zeros(ldim)
        for i in np.arange(self.n):
            pertubation = np.array([[random.uniform(-1, 1) for j in np.arange(ldim)]])
            c_plus = self.c0 + pertubation
            c_minus = self.c0 - pertubation
            self.delta_y[i] = self._rate_vector(c_plus) - self._rate_vector(c_minus)
            delta_c0 += self.delta_y[i] * pertubation.squeeze() / np.linalg.norm(pertubation.squeeze(), 2)
        
        self.c0 += self.learning_rate * delta_c0

    def _rate_vector(self, c):
        decoded = self.generator.decoder.predict(c)
        # return mse(self.y_true, decoded[0])
        # return -mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return 1/mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return ssmi(self.y_true.squeeze(), decoded[0].squeeze())
        return -1/ssmi(self.y_true.squeeze(), decoded[0].squeeze())

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Optimizer parameters')

    parser.add_argument('--model', help='folder containing parameters and weights of the model', default="models\\batch-size_16#epochs_50#grayscale_True#recon-loss_mse#heading-weight_0#test")
    parser.add_argument('--trials', help='number of trials the optimizer should run for', default="10")
    parser.add_argument('--sample-points', help='how many sample points are used for gradients', default="50")
    parser.add_argument('--learning-rate', help='weight for the update of the point in latent space', default="1")
    parser.add_argument('--search-radius', help='tbd', default="0.5")
    parser.add_argument('--save-intervall', help='after how many interations a video is saved', default="50")

    parser.add_argument('--prefix', help='prefix of files for sorting', default=None)

    args = parser.parse_args()

    model = args.model

    trials = int(args.trials)
    sample_points = int(args.sample_points)
    learning_rate = float(args.learning_rate)
    search_radius = float(args.search_radius)
    save_intervall = int(args.save_intervall)

    iterations = trials//(sample_points * 2)

    prefix = args.prefix

    generator = VAE.load(model)

    batch_size = 1
    video_dim = generator.input_shape

    dataset = "20220930-134704_1.csv" # trainingset
    # dataset = "20221110-174245_1_ws.csv"

    data = SelfmotionDataGenerator(f"/mnt/masc_home/kressal/datasets/selfmotion/{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    # data = SelfmotionDataGenerator(f"N:\\Datasets\\selfmotion\\{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    y_true = data[0][0][0]
    reconstructed_images, latent_representations, predicted_heading = generator.reconstruct(np.expand_dims(y_true, 0))

    optimizer = FDGDOptimizer(sample_points, generator, learning_rate, search_radius, ground_truth=y_true)

    rating = np.zeros(iterations)
    distance = np.zeros(iterations)
    heading_error = np.zeros(iterations)
    embedding_error = np.zeros(iterations)

    x = np.arange(iterations) * sample_points * 2

    t = tqdm(np.arange(iterations))
    for i in t:
        optimizer.step()
        current_guess = optimizer.generator.decoder.predict(np.expand_dims(optimizer.c0, 0))
        predicted_heading = optimizer.generator.heading_decoder.predict(np.expand_dims(optimizer.c0, 0))
        rating[i] = ssmi(y_true.squeeze(), current_guess.squeeze())
        distance[i] = np.linalg.norm(optimizer.c0-latent_representations, 2)
        embedding_error[i] = mse(latent_representations, optimizer.c0)
        heading_error[i] = mse(data[0][1][1], predicted_heading)
        t.set_description(f"rating: {rating[i]:.4f}, mean delta_y: {optimizer.delta_y.mean():.4f}", refresh=True)
        if i%save_intervall == 0:
            save_video(f"optimizer_results/{prefix}__it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}#rating_{rating[i]:.4f}_guess.mp4", current_guess[0]*255)

    save_video(f"optimizer_results/{prefix}__it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}#rating_{rating[i]:.4f}_guess.mp4", current_guess[0]*255)

    output_name = f"{prefix}__it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}#ld_{generator.latent_space_dim}"
    figsize = 12
    fontsize = 15
    plt.figure(figsize=(figsize, figsize))
    plt.plot(x, rating)
    plt.ylabel("SSMI", size=fontsize)
    plt.xlabel("Trial", size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.title(f"{output_name}", size=fontsize)
    plt.savefig(f"optimizer_results/{output_name}_rating.png")

    plt.figure(figsize=(figsize, figsize))
    plt.plot(x, distance)
    plt.ylabel("L2 Norm", size=fontsize)
    plt.xlabel("Trial", size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.title(f"{output_name} - distance", size=fontsize)
    plt.savefig(f"optimizer_results/{output_name}_distance.png")

    plt.figure(figsize=(figsize, figsize))
    plt.plot(x, heading_error)
    plt.ylabel("MSE", size=fontsize)
    plt.xlabel("Trial", size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.title(f"{output_name} - MSE(heading)", size=fontsize)
    plt.savefig(f"optimizer_results/{output_name}_heading_error.png")

    plt.figure(figsize=(figsize, figsize))
    plt.plot(x, embedding_error)
    plt.ylabel("MSE", size=fontsize)
    plt.xlabel("Trial", size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.title(f"{output_name} - MSE(embedding)", size=fontsize)
    plt.savefig(f"optimizer_results/{output_name}_embedding_error.png")

    plt.figure(figsize=(figsize, figsize))
    plt.boxplot(optimizer.c0)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.title(f"{output_name} - embedding", size=fontsize)
    plt.savefig(f"optimizer_results/{output_name}_embedding_box.png")

    # output_name = f"it_{trials}#_sp_{sample_points}"
    save_video(f"optimizer_results/{prefix}__it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}_original.mp4", y_true*255)
    save_video(f"optimizer_results/{prefix}__it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}_reconstructed.mp4", reconstructed_images[0]*255)
    pass