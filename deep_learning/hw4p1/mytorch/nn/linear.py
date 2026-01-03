import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)
        self.out_features = out_features
        self.in_features = in_features


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)

        Handles arbitrary batch dimensions like PyTorch
        """

        # Store input for backward pass
        self.A = A
        self.N = A.shape[0]

        Z = self.A @ self.W.T + self.b.T

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Compute gradients (refer to the equations in the writeup)
        dLdZ = np.reshape(dLdZ, (-1, self.out_features))
        A = np.reshape(self.A, (-1, self.in_features))
        self.dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ A
        self.dLdb = dLdZ.sum(axis=0)

        # Return gradient of loss wrt input
        return np.reshape(self.dLdA, self.A.shape)
