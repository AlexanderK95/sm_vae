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
    def __init__(self, sample_size:int=50, gegnerator_model=None, learning_rate=0.1, search_radius=0.1, ground_truth=None):
        self.n = sample_size
        self.generator = VAE.load(gegnerator_model)
        self.sigma = search_radius
        self.learning_rate = learning_rate

        self.c0 = np.array([random.uniform(-2, 2) for j in range(int(self.generator.latent_space_dim))])

        self.y_true = ground_truth


    def step(self):
        ldim = self.generator.latent_space_dim
        pertubations = np.zeros([self.n, ldim])
        # pertubations = np.zeros([[random.uniform(-2, 2) for j in range(int(ldim))] for i in range(num_samples)])
        self.delta_y = np.zeros(self.n)
        delta_c0 = np.zeros(ldim)
        for i in np.arange(self.n):
            pertubation = np.array([[random.uniform(-2, 2) for j in np.arange(ldim)]])
            pertubations[i] = self.c0 + pertubation
            antithetic_pertubation = self.c0 - pertubation
            self.delta_y[i] = self._rate_vector(np.expand_dims(pertubations[i], 0)) - self._rate_vector(antithetic_pertubation)
            delta_c0 += self.delta_y[i] * pertubations[i] / np.linalg.norm(pertubations[i], 2)
        
        self.c0 = self.c0 * (1 + self.learning_rate * delta_c0)

    def _rate_vector(self, c):
        decoded = self.generator.decoder.predict(c)
        # return mse(self.y_true, decoded[0])
        return ssmi(self.y_true.squeeze(), decoded[0].squeeze())

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Optimizer parameters')

    parser.add_argument('--model', help='folder containing parameters and weights of the model', default="models\\batch-size_16#epochs_50#grayscale_True#recon-loss_mse#heading-weight_0#test")
    parser.add_argument('--iterations', help='number of iterations the optimizer should run for', default="200")
    parser.add_argument('--sample-points', help='how many sample points are used for gradients', default="50")
    parser.add_argument('--learning-rate', help='weight for the update of the point in latent space', default="1")
    parser.add_argument('--search-radius', help='tbd', default="0.5")
    parser.add_argument('--save-intervall', help='after how many interations a video is saved', default="50")

    args = parser.parse_args()

    model = "models/batch-size_16#epochs_2000#grayscale_True#recon-loss_binary_crossentropy#heading-weight_0.2#kl-weight_0.00045#latent-dim_360"

    iterations = int(args.iterations)
    sample_points = int(args.sample_points)
    learning_rate = float(args.learning_rate)
    search_radius = float(args.search_radius)
    save_intervall = int(args.save_intervall)

    batch_size = 2
    video_dim = [8, 256, 256]

    data = SelfmotionDataGenerator("N:/Datasets/selfmotion/20220930-134704_1.csv", batch_size, video_dim, grayscale=True, shuffle=True)
    y_true = data[0][0][0]

    optimizer = FDGDOptimizer(sample_points, model, ground_truth=y_true)

    rating = np.zeros(iterations)

    t = tqdm(np.arange(iterations))
    for i in t:
        optimizer.step()
        current_guess = optimizer.generator.decoder.predict(np.expand_dims(optimizer.c0, 0))
        rating[i] = ssmi(y_true.squeeze(), current_guess.squeeze())
        t.set_description(f"rating: {rating[i]:.4f}, mean delta_y: {optimizer.delta_y.mean():.4f}", refresh=True)
        if i%save_intervall == 0:
            save_video(f"optimizer_results/it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}_guess.mp4", current_guess[0]*255)

    save_video(f"optimizer_results/it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}_guess.mp4", current_guess[0]*255)

    output_name = f"it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}"
    figsize = 12
    plt.figure(figsize=(figsize, figsize))
    plt.plot(rating)
    plt.xlabel("Rating", size=20)
    plt.ylabel("Iteration", size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f"{output_name}", size=10)
    plt.savefig(f"results/{output_name}_rating.png")

    # output_name = f"it_{iterations}#_sp_{sample_points}"
    save_video(f"optimizer_results/it_{i}#sp_{sample_points}#lr_{learning_rate}#sr_{search_radius}_original.mp4", y_true*255)
    # save_video(f"optimizer_results/{output_name}_guess.mp4", current_guess[0]*255)
    pass