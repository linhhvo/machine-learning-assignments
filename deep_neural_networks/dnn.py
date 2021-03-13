#######################################################################################################################
# Program:  dnn.py
# Desc:     This program implements a multi-layer neural network classifier
# Date:     Mar 2021
# Note:     The code is adapted from Prof. Andrew Ng's Neural Networks and Deep Learning course at
#           https://github.com/enggen/Deep-Learning-Coursera
#######################################################################################################################
import numpy as np
from utils_actfunc import *

class Layer:
    '''
    A Layer object represents a single hidden layer with n_h hidden nodes
    '''

    def __init__(self, n_h, activation):
        '''
        This method is called when you create a Layer object
        Arguments:
            n_h -- number of hidden nodes in this layer
            activation -- the name of the activation function used for this layer (either 'relu' or 'sigmoid')
        '''
        self.n_h = n_h
        self.act_function = activation

    def __str__(self):
        '''
        You can print out the content of this layer if you want to make sure everything is correctly set up
        '''
        result = "Number of hidden nodes: " + str(self.n_h) + "\nShape of Weights:\t" \
                 + str(self.W.shape) + "\nShape of Biases:\t" + str(self.b.shape)

        return result

    def get_parameters(self):
        return self.W, self.b

    def get_weights(self):
        return self.W

    def get_biases(self):
        return self.b

    def initialize(self, n_h_prev):
        """
        Initialize the weights and biases for this layer

        Argument:
        n_h_prev -- number of hidden nodes in the previous layer
        """

        # W -- initialized vector of shape (n_h_prev, n_h)
        # b -- initialized vector of shape(n_h, 1)
        # n_h -- number of hidden nodes in this layer
        self.W = np.random.randn(n_h_prev, self.n_h)*0.01
        self.b = np.zeros(shape = (self.n_h, 1))


    def forward(self, A_prev):
        """
        Implement the forward propagation including the linear combination and nonlinear activation

        Argument:
        A_prev -- activations from previous layer (or input data) of shape (n_h_prev, m), where m is the number of instances

        Return:
        A -- the activations in this layer
        """
        # Start your code here
            # Find Z: one line
            # Find A: about four lines (Hint: you will need to call either the sigmoid() or relu() functions in utils_actfunc.py)

        # End your code here

        return self.A

    def backward(self, dA, A_prev):
        """
        Implement the backward propagation to find dW and db for this layer

        Arguments:
        dA -- post-activation gradient for this layer, it is the derivative of the total cost with respect to A
        """

        if self.act_function == "relu":
            dZ = relu_backward(dA, self.Z)
        elif self.act_function == "sigmoid":
            dZ = sigmoid_backward(dA, self.Z)

        # Start your code here: four lines
        #Find the following:
            #m -- the number of instances in the dataset
            #dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            #dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            #db -- Gradient of the cost with respect to b (current layer l), same shape as b


        # End your code here

        return dA_prev

    def update(self, learning_rate):
        """
        Update parameters using gradient descent
        """
        # Start your code here: two lines

        # End your code here

class DNNClassifier():
    '''
    A DNNClassifier assembles L hidden layers together and performs the classification task
    '''
    def __init__(self, layers = None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.costs = []

    def add(self, layer):
        '''
        Add a single layer to the network

        Arguments:
        layer -- a hidden layer object with one or more hidden nodes
        '''

        # Start your code here: one line

        # End your code here

    def add_input(self, X_train):
        '''
        Add the input data to layer 0 and initialize all other hidden layers

        Arguments:
        X_train -- training data of shape (n_x, m), where n_x is the number of features and m is the number of instances
        '''

        # Start your code here: four lines
            # Create a Layer object (layer_0) to hold the input (the n_h for this layer is X_train.shape[0]), the activation function is None
            # Set layer_0's activation (A) to be X_train
            # Insert layer_0 into the front of the layers list
            # Find the value of L. Note that here L = # hiddlen layers + 1
        # End your code here

        # Initialize weights and biases for each layer
        # Start your code here: two lines

        # End your code here

    def forward_prop(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- input data, numpy array of shape (n_x, m), where n_x is the number of features and m is the number of instances

        Returns:
        aL -- last post-activation value

        """

        L = len(self.layers)       # Note that here L = # hiddlen layers + 1

        # Forward propagate through layers 1 to L-1
        A_prev = X
        for i in range(1, L):
            # Start your code here: two lines

            # End your code here

        aL = A      # aL is the activation of the last layer, shape (1, m)

        return aL

    def compute_cost(self, aL, y_train):
        """
        Implement the cost function

        Arguments:
        aL -- probability vector corresponding to your label predictions, shape (1, m)
        Y -- true "label" vector, shape (1, m)

        Returns:
        cost -- cross-entropy cost
        """
        m = y_train.shape[1]

        # Compute loss from AL and y.
        cost = -np.sum(np.multiply(y_train, np.log(aL)) + np.multiply(1 - y_train, np.log(1 - aL)))/m

        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17), i.e., convert it into a single number
        cost = np.squeeze(cost)

        return cost

    def back_prop(self, aL, y_train):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        aL -- the probability vector, output of the forward propagation
        y_train -- the true "label" vector

        """

        L = len(self.layers)
        dAL = - (np.divide(y_train, aL) - np.divide(1 - y_train, 1 - aL)) # derivative of the cost with respect to aL

        dA = dAL
        for i in range(L-1, 0, -1):
            # Start your code here: two lines


            # End your code here


    def update_parameters(self, learning_rate):
        L = len(self.layers)
        for i in range(1, L):
            # Start your code here: one line

            # End your code here

    def fit(self, X_train, y_train, learning_rate = 0.01, num_epochs = 1000, print_cost = False):
        '''
        Fit the data by running the propagation for a number of epochs

        Arguments:
        learning_rate -- the learning rate
        num_epochs -- number of iterations for training
        print_cost -- whether to print the loss, True to print the loss every 100 steps
        '''

        # Add the input to the network
        self.add_input(X_train)

        for i in range(num_epochs):
            # Start your code here: four lines
                # forward prop
                # compute the cost for this epoch
                # back prop
                # update parameters (gradient descent)





            # End your code here

            # store and print the cost every 100 epochs
            if i % 100 == 0:
                self.costs.append(cost)
                if print_cost:
                    print("Cost after iteration {}: {}".format(i, cost))


    def predict(self, X_test):
        '''
        Predict whether the label is 0 or 1 using the learned model

        Arguments:
        X_test -- test data of size (n_x, number of testing instances)

        Returns:
        y_prediction -- a numpy array (vector) containing all predictions (0/1) for instances in X_test
        '''
        m = X_test.shape[1]
        y_prediction = np.zeros((1, m))

        # Start your code here: one line
            # Find the aL for X_test: one line
            # Convert probabilities a[0,i] to actual predictions p[0,i]: two lines


        # End your code here

        return y_prediction
