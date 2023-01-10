import numpy as np
import random
from optimizer.losses import mse, ssmi

class NSEOptimizer:
    def __init__(self, n_pop:int=20, top_n = 5, n_iter=10, generator=None, search_radius=1, min_error_weight = 0.01, error_weight = 1, decay_rate = 0.95, ground_truth=None):
        '''
        Arguments : model — Model object(single layer neural network here),
        x — numpy array of shape (batch size, number of features),
        y — numpy array of shape (batch size, number of classes),
        top_n — Number of elite parameters to consider for calculating the
        best parameter by taking mean
        n_pop — Population size of the parameters
        n_iter — Number of iteration 
        sigma_error — The standard deviation of errors while creating 
        population from best parameter
        error_weight — Contribution of error for considering new population
        decay_rate — Rate at which the weight of the error will reduce after 
        each iteration, so that we don't deviate away at the 
        point of convergence. It controls the balance between 
        exploration and exploitation

        Returns : Model object with updated parameters/weights
        '''

        self.n = n_pop
        self.n_iter = n_iter
        self.top_n = top_n
        self.generator = generator
        self.sigma_error = search_radius
        self.min_error_weight = min_error_weight
        self.error_weight = error_weight
        self.decay_rate = decay_rate

        self.c0 = np.array([random.uniform(-2, 2) for j in range(int(self.generator.latent_space_dim))])

        self.population = np.array([[random.uniform(-2, 2) for j in range(int(self.generator.latent_space_dim))] for i in range(self.n)])
        self.best_guess = self.population[0]


        self.y_true = ground_truth


    def step(self):
        ldim = self.generator.latent_space_dim
        # pertubations = np.zeros([self.n, ldim])
        # pertubations = np.zeros([[random.uniform(-2, 2) for j in range(int(ldim))] for i in range(num_samples)])
        self.delta_y = np.zeros(self.n)
        self.evaluation_values = np.zeros(self.n)
        delta_c0 = np.zeros(ldim)

        for i in range(self.n_iter):
            # Generating the population of parameters
            self.population = [self.best_guess + self.error_weight * self.sigma_error * np.random.randn(ldim) for i in range(self.n)]

        for i in np.arange(self.n):
            self.evaluation_values[i] = self._rate_vector(np.expand_dims(self.population[i], 0))

        eval_list = zip(self.evaluation_values, self.population)
        eval_list = sorted(eval_list, key = lambda x: x[0], reverse = True)
        evaluation_values, self.population = zip(*eval_list)

        # Taking the mean of the elite parameters
        self.best_guess = np.stack(self.population[:self.top_n], axis=0).mean(axis=0)

        #Decaying the weight
        self.error_weight = max(self.error_weight*self.decay_rate, self.min_error_weight)
        
        # self.c0 += self.learning_rate * delta_c0


    def _rate_vector(self, c):
        decoded = self.generator.decoder.predict(c)
        # heading = self.generator.heading_decoder(c)
        # return mse(self.y_true, decoded[0])
        # return -mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return 1/mse(self.y_true.squeeze(), decoded[0].squeeze())
        # return ssmi(self.y_true.squeeze(), decoded[0].squeeze())
        return ssmi(self.y_true.squeeze(), decoded[0].squeeze())
        # return -mse(self.y_true.squeeze(), heading.numpy().squeeze())


