"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
from typing import Callable, List, Tuple
from neuralnet import INIT_FN_TYPE, Linear, Sigmoid, SoftMaxCrossEntropy, shuffle

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.


class NN2:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight
                               matrices for layer. This function takes in a
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.z1_sigmoid = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.z2_sigmoid = Sigmoid()
        self.linear3 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.y_J_softmax = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation.
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # call forward pass for each layer
        a = self.linear1.forward(x)
        z1 = self.z1_sigmoid.forward(a)
        b = self.linear2.forward(z1)
        z2 = self.z2_sigmoid.forward(b)
        c = self.linear3.forward(z2)
        return self.y_J_softmax.forward(c, y)

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # call backward pass for each layer
        gJ = 1
        gc = gJ * self.y_J_softmax.backward(y, y_hat)
        gz2 = self.linear3.backward(gc)
        gb = self.z_sigmoid.backward(gz2)
        gz1 = self.linear2.backward(gb)
        ga = self.z_sigmoid.backward(gz1)
        self.linear1.backward(ga)
        
    def step(self):
        """
        Apply SGD update to weights.
        """

        self.linear1.step()
        self.linear2.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # compute loss over the entire dataset
        #  Hint: reuse your forward function
        
        losses = []
        for i in range(0, y.size):
            (_, xi_loss) = self.forward(X[i], y[i])
            losses.append(xi_loss)
            
        return np.mean(losses)


    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_train: train data
        :param y_train: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # train network
        
        train_losses = np.zeros(n_epochs)
        test_losses = np.zeros(n_epochs)

        for epoch in range(0, n_epochs):
            X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train, epoch)

            for i in range(0, y_train.size):
                (y_hat, _) = self.forward(X_train_shuffled[i], y_train_shuffled[i])
                self.backward(y_train_shuffled[i], y_hat)
                self.step()

            train_losses[epoch] = self.compute_loss(X_train_shuffled, y_train_shuffled)
            test_losses[epoch] = self.compute_loss(X_test, y_test)

        return (train_losses, test_losses)

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # make predictions and compute error
        
        predicted_labels = []
        error_count = 0
        for i in range(0, y.size):
            (y_hat, _) = self.forward(X[i], y[i])
            label = np.argmax(y_hat)
            
            predicted_labels.insert(i, label)

            if (label != y[i]):
                error_count += 1
            
        return (predicted_labels, error_count / y.size)