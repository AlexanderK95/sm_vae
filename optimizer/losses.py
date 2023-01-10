import numpy as np
from skimage.metrics import structural_similarity


def mse(Y_true, Y_pred):
    return np.square(np.subtract(Y_true,Y_pred)).mean()

def ssmi(Y_true, Y_pred):
    index = 0
    for i in np.arange(Y_true.shape[0]):
        index += structural_similarity(Y_true[i], Y_pred[i])
    index = index / Y_true.shape[0]
    return index