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
from optimizer.FDGDOptimizer import FDGDOptimizer
from optimizer.NSEOptimizer import NSEOptimizer
from optimizer.GeneticOptimizer import GeneticOptimizer
from optimizer.losses import mse, ssmi

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Optimizer parameters')

    parser.add_argument('--model', help='folder containing parameters and weights of the model', default="models\\batch-size_32#epochs_2000#grayscale_True#recon-loss_binary_crossentropy#heading-weight_0.2#kl-weight_0.00045#latent-dim_210#learning-rate_0.0002")
    parser.add_argument('--trials', help='number of trials the optimizer should run for', default="6000")
    parser.add_argument('--sample-points', help='how many sample points are used for gradients', default="25")
    parser.add_argument('--learning-rate', help='weight for the update of the point in latent space', default="1")
    parser.add_argument('--search-radius', help='tbd', default="0.5")
    parser.add_argument('--save-intervall', help='after how many interations a video is saved', default="50")

    parser.add_argument('--prefix', help='prefix of files for sorting', default="")

    args = parser.parse_args()

    model = args.model

    trials = int(args.trials)
    sample_points = int(args.sample_points)
    learning_rate = float(args.learning_rate)
    search_radius = float(args.search_radius)
    save_intervall = int(args.save_intervall)

    iterations = trials//(sample_points * 2)

    prefix = args.prefix
    prefix = "nse"

    generator = VAE.load(model)

    batch_size = 1
    video_dim = generator.input_shape

    dataset = "20220930-134704_1.csv" # trainingset
    # dataset = "20221110-174245_1_ws.csv"

    # data = SelfmotionDataGenerator(f"/mnt/masc_home/kressal/datasets/selfmotion/{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    data = SelfmotionDataGenerator(f"N:\\Datasets\\selfmotion\\{dataset}", batch_size, video_dim, grayscale=True, shuffle=True)
    y_true = data[0][0][0]
    # y_true = data[0][1][0]
    reconstructed_images, latent_representations, predicted_heading = generator.reconstruct(np.expand_dims(y_true, 0))


    # optimizer = GeneticOptimizer(50, generator, ground_truth=y_true, selectivity=0.8, mutation_rate=0.2, mutation_size=0.1, heritability=0.8)
    # optimizer = FDGDOptimizer(sample_points, generator, learning_rate, search_radius, ground_truth=y_true)
    optimizer = NSEOptimizer(generator=generator, ground_truth=y_true, n_pop=50, top_n=5, n_iter=20, search_radius=5, error_weight=2)

    # optimizer.step()

    # optimizer = FDGDOptimizer(sample_points, generator, learning_rate, search_radius, ground_truth=predicted_heading)
    

    rating = np.zeros(iterations)
    distance = np.zeros(iterations)
    heading_error = np.zeros(iterations)
    embedding_error = np.zeros(iterations)

    x = np.arange(iterations) * sample_points * 2

    t = tqdm(np.arange(iterations))
    # t = np.arange(iterations)
    for i in t:
        # print(f"{i}/{t.size}")
        optimizer.step()
        current_guess = optimizer.generator.decoder.predict(np.expand_dims(optimizer.c0, 0))
        predicted_heading = optimizer.generator.heading_decoder.predict(np.expand_dims(optimizer.c0, 0))
        rating[i] = ssmi(y_true.squeeze(), current_guess.squeeze())
        distance[i] = np.linalg.norm(optimizer.c0-latent_representations, 2)
        embedding_error[i] = mse(latent_representations, optimizer.c0)
        heading_error[i] = mse(data[0][1][1], predicted_heading)
        # t.set_description(f"rating: {rating[i]:.4f}, mean delta_y: {optimizer.delta_y.mean():.4f}", refresh=True)
        t.set_description(f"rating: {rating[i]:.4f}", refresh=True)
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