import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.output_height = A.shape[3] - self.kernel + 1
        self.output_width = A.shape[2] - self.kernel + 1

        Z = np.zeros((A.shape[0], A.shape[1], self.output_height, self.output_width))
        self.max_indices = np.zeros_like(Z, dtype=int)

        for i in range(self.output_height):
            for j in range(self.output_width):
                Z[:, :, i, j] += np.max(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

                flattened_kernel_segment = A[:, :, i:i+self.kernel, j:j+self.kernel].reshape(A.shape[0], A.shape[1], -1)
                self.max_indices[:, :, i, j] = np.argmax(flattened_kernel_segment, axis=2)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #coordinates = np.unravel_index(self.max_indices, self.A.shape)
        dLdA = np.zeros_like(self.A)

        for i in range(self.output_height):
            for j in range(self.output_width):
                indexing_flaten = self.max_indices[:, :, i, j]
                relative_height, relative_width = np.unravel_index(indexing_flaten, (self.kernel, self.kernel))
                abs_height = relative_height + i
                abs_width = relative_width + j

                for b in range(dLdZ.shape[0]):
                    for c in range(dLdZ.shape[1]):
                        dLdA[b, c, abs_height[b, c], abs_width[b, c]] += dLdZ[b, c, i, j]
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.output_height = A.shape[2] - self.kernel + 1
        self.output_width = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                Z[:,:, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)
        for b in range(dLdZ.shape[0]):
            for c in range(dLdZ.shape[1]):
                for i in range(dLdZ.shape[2]):
                    for j in range(dLdZ.shape[3]):
                        dLdA[b, c, i:i+self.kernel, j:j+self.kernel] += dLdZ[b, c, i, j] / (self.kernel * self.kernel)

        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        pooled_A = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled_A)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        upsampled_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(upsampled_dLdZ)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooled_A = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(pooled_A)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        upsampled_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(upsampled_dLdZ)
        return dLdA
