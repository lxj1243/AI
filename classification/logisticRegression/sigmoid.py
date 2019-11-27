import numpy as np


def sigmoid(z):
    """

    :param z: theta.T*X
    :return:
    """
    y = 1/(1+np.exp(-z))
    return y
