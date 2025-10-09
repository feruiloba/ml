# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        # You are given a 128×24 input (128 time steps, with a 24-dimensional vector at each time)
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, padding=0, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, padding=0, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, padding=0, stride=1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()
        ] # Add the layers in the correct order

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        #  For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)

        self.conv1.conv1d_stride1.W = w1.T.reshape(self.conv1.out_channels, self.conv1.kernel_size, self.conv1.in_channels).transpose(0,2,1)
        self.conv2.conv1d_stride1.W = w2.T.reshape(self.conv2.out_channels, self.conv2.kernel_size, self.conv2.in_channels).transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w3.T.reshape(self.conv3.out_channels, self.conv3.kernel_size, self.conv3.in_channels).transpose(0,2,1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method
        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(in_channels=24, out_channels=8, kernel_size=8, padding=0, stride=4)
        self.conv2 = Conv1d(in_channels=8, out_channels=16, kernel_size=1, padding=0, stride=1)
        self.conv3 = Conv1d(in_channels=16, out_channels=4, kernel_size=1, padding=0, stride=1)
        self.layers = [
            self.conv1,
            ReLU(),
            self.conv2,
            ReLU(),
            self.conv3,
            Flatten()
        ] # Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)
        #   4 : Slice the weight matrix and reduce to only the shared weights
        #   (hint: be careful, steps 1-3 are similar, but not exactly like in the simple scanning MLP)

        self.conv1.conv1d_stride1.W = w1.T.reshape(self.conv1.out_channels, self.conv1.kernel_size, self.conv1.in_channels).transpose(0,2,1)
        self.conv2.conv1d_stride1.W = w2.T.reshape(self.conv2.out_channels, self.conv2.kernel_size, self.conv2.in_channels).transpose(0,2,1)
        self.conv3.conv1d_stride1.W = w2.T.reshape(self.conv2.out_channels, self.conv2.kernel_size, self.conv2.in_channels).transpose(0,2,1)

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
