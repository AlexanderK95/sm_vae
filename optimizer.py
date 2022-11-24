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


class GeneticOptimizer:
    def __init__(self, population_size:int=50, generator=None, selectivity=0.5, heritability=0.5, mutation_rate=0.05, mutation_size=0.1, ground_truth=None):
        self.n = population_size
        self.s = selectivity
        self.h = heritability
        self.r = mutation_rate
        self.sigma = mutation_size

        self.generator = generator

        self.population = np.array([[random.uniform(-2, 2) for j in range(int(self.generator.latent_space_dim))] for i in range(self.n)])

        self.y_true = ground_truth    

        self.indices = np.arange(self.generator.latent_space_dim)

    def step(self):
        decoded = self.generator.decoder.predict(self.population)
        vf = np.zeros(self.n)
        for i in np.arange(decoded.shape[0]):
            vf[i] = -1/ssmi(self.y_true.squeeze(), decoded[i].squeeze())
        p = self._get_probabilities(vf)
        for i in np.arange(self.n):
            parents_idx = np.random.choice(np.arange(self.n), 2, True, p)
            idx_p1 = np.random.choice(self.indices, int(self.n * self.h), False)
            mask = np.array([(i in idx_p1) for i in self.indices])
            idx_p2 = self.indices[~mask]
            new_latent = np.zeros_like(self.indices)
            new_latent[idx_p1] = self.population[parents_idx[0]][idx_p1]
            new_latent[idx_p2] = self.population[parents_idx[1]][idx_p2]
            self.population[i] = new_latent
        #Todo add Mutation
        pass

    def _get_probabilities(self, fitness_vector):
        f_min = fitness_vector.min()
        f_std = fitness_vector.std()
        k = f_std/self.s
        w = np.exp((fitness_vector-f_min)/k)
        return w/np.sum(w)


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

    # data = SelfmotionDataGenerator(f"/mnt/masc_home/kressal/datasets/selfmotion/{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    data = SelfmotionDataGenerator(f"N:\\Datasets\\selfmotion\\{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    y_true = data[0][0][0]
    reconstructed_images, latent_representations, predicted_heading = generator.reconstruct(np.expand_dims(y_true, 0))


    optimizer = GeneticOptimizer(sample_points, generator, ground_truth=y_true)
    optimizer.step()
    optimizer.step()
    optimizer.step()
    optimizer.step()
    optimizer.step()

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