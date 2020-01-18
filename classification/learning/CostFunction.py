import numpy as np

import logisticRegression as lr
# from logisticRegression.sigmoid import sigmoid
# 像上面那样导入，就可以直接sigmoid(x)这样使用


def CFlogisticregression(theta, X, Y):
    """
    CF means cost function
    :param theta:
    :param X:
    :param Y:
    :return:
    """

    theta = np.ndarray(theta)
    X = np.ndarray(X)
    Y = np.ndarray(Y)

    # y=1时的代价函数
    firstpart = np.multiply(-Y,np.log(lr.sigmoid(np.multiply(X,theta.T))))
    # firsrtpart = -Y*np.log(sigmoid(theta.T*X)

    # y=0时的代价函数
    secondpart = np.multiply(1-Y,np.log(1-lr.sigmoid(np.multiply(X,theta.T))))
    # secondpart = (1-Y)*np.log(1-sigmoid(theta.T*X)
    return
