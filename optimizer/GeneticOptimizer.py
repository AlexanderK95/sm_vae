import numpy as np
import random
from optimizer.losses import mse, ssmi

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

        self.c0 = np.zeros(self.generator.latent_space_dim)

    def step(self):
        decoded = self.generator.decoder.predict(self.population)
        mutation_candidates = random.sample(list(range(self.n)), int(self.n*self.r))
        vf = np.zeros(self.n)
        for i in np.arange(decoded.shape[0]):
            vf[i] = ssmi(self.y_true.squeeze(), decoded[i].squeeze())
        p = self._get_probabilities(vf)
        self.c0 = self.population[np.argmax(p)]
        for i in np.arange(self.n):
            parents_idx = np.random.choice(np.arange(self.n), 2, True, p)
            idx_p1 = np.random.choice(self.indices, int(self.n * self.h), False)
            mask = np.array([(i in idx_p1) for i in self.indices])
            idx_p2 = self.indices[~mask]
            new_latent = np.zeros_like(self.indices)
            new_latent[idx_p1] = self.population[parents_idx[0]][idx_p1]
            new_latent[idx_p2] = self.population[parents_idx[1]][idx_p2]
            self.population[i] = new_latent
            if i in mutation_candidates:
                mutation = np.random.multivariate_normal(np.zeros_like(self.indices), np.eye(self.indices.size)*self.sigma, 1).squeeze()
                self.population[i] += mutation

    def _get_probabilities(self, fitness_vector):
        f_min = fitness_vector.min()
        f_std = fitness_vector.std()
        k = f_std/self.s
        w = np.exp((fitness_vector-f_min)/k)
        return w/np.sum(w)