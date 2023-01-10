import numpy as np
import random
from optimizer.losses import mse, ssmi

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
        # heading = self.generator.heading_decoder(c)
        # return mse(self.y_true, decoded[0])
        # return -mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return 1/mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return ssmi(self.y_true.squeeze(), decoded[0].squeeze())
        return -1/ssmi(self.y_true.squeeze(), decoded[0].squeeze())
        # return -mse(self.y_true.squeeze(), heading.numpy().squeeze())