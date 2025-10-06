import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features), dtype=np.float32) # (C1, C0)
        self.b = np.zeros((out_features, 1), dtype=np.float32) # (N, 1)

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A
        self.N = A.shape[0] # store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        self.ones = np.ones((self.N, 1))

        # A is (N, C0), W^T is (C0, C1), so A*W^T is (N, C1)
        Z = self.A @ self.W.T + self.ones @ self.b.T
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.ones

        if self.debug:
            self.dLdA = dLdA

        return dLdA
