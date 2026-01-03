import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        # Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        reduced_Z = Z - np.max(Z, axis=self.dim, keepdims=True)
        sum_exp_reduced_Z = np.sum(np.exp(reduced_Z), axis=self.dim, keepdims=True)
        self.A = np.exp(reduced_Z) / sum_exp_reduced_Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Implement backward pass

        # Get the shape of the input
        original_shape = self.A.shape
        shape = list(range(len(self.A.shape)))

        # Find the dimension along which softmax was applied
        C = original_shape[self.dim]
        shape.append(shape.pop(self.dim))

        # Reshape input to 2D
        if len(shape) > 2:
            A_transposed = np.transpose(self.A, shape)
            self.A = A_transposed.reshape(-1, C)
            dLdA = np.transpose(dLdA, shape)
            dLdA = dLdA.reshape(-1, C)

        N = self.A.shape[0]

        # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
        dLdZ = np.zeros_like(dLdA)

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            # Initialize the Jacobian with all zeros.
            J = np.zeros((C, C))

            # Fill the Jacobian matrix, please read the writeup for the conditions.
            for m in range(C):
                for n in range(C):
                    J[m, n] = self.A[i, m] * (1 - self.A[i, m]) if m == n else -self.A[i, m] * self.A[i, n]

            # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
            # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
            dLdZ[i, :] = dLdA[i] @ J


        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(A_transposed.shape)
            self.A = np.transpose(self.A, np.argsort(shape))
            dLdZ = dLdZ.reshape(A_transposed.shape)
            dLdZ = np.transpose(dLdZ, np.argsort(shape))

        return dLdZ


