import numpy as np
import logisticRegression as lr

def costRegu(theta, X, y, learningRate):
    """

    :param theta:
    :param X:
    :param y:
    :param learningRate:
    :return:
    """
    theta=np.ndarray(theta)
    X=np.ndarray(X)
    y=np.ndarray(y)
    firstpart = np.multiply(-y,np.log(lr.sigmoid(np.multiply(X,theta.T))))
    secondpart = np.multiply(1 -y, np.log(1 - lr.sigmoid(np.multiply(X, theta.T))))
    regu=(learningRate/(2*len(X))*np.sum(np.power(theta[:, 1:theta.shape[1]], 2)))
    return np.sum(firstpart-secondpart)/(len(X))+regu
