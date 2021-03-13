#########################################################################
# Program:  utils_actfunc (activation functions)
# Source:   https://github.com/enggen/Deep-Learning-Coursera
#########################################################################
import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """

    return 1 / (1 + np.exp(-Z))


def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    return np.maximum(0, Z)


def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ
