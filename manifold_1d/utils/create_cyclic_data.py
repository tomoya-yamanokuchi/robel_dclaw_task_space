import numpy as np


def create_cyclic_data(x):
    return np.concatenate([x, x[:1]], axis=0)
