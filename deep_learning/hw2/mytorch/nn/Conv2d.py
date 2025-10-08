import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        self.output_height = A.shape[2] - self.kernel_size + 1
        self.output_width = A.shape[3] - self.kernel_size + 1

        Z = np.zeros((A.shape[0], self.out_channels, self.output_height, self.output_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                kernel_out = np.tensordot(A[:, :, i:i+self.kernel_size, j:j+self.kernel_size], self.W, axes=((1, 2, 3), (1, 2, 3))) + self.b
                Z[:,:, i, j] = kernel_out

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # dLdW
        dLdZ_height = dLdZ.shape[2]
        dLdZ_width = dLdZ.shape[3]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dLdW_ij = np.tensordot(dLdZ, self.A[:, :, i:i+dLdZ_height, j:j+dLdZ_width], axes=((0, 2, 3), (0, 2, 3)))
                self.dLdW[:, :, i, j] = dLdW_ij

        # dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # dLdA
        padding = int(self.kernel_size - 1)
        dLdZ_padded = np.pad(dLdZ, ((0,0),(0,0),(padding,padding), (padding,padding)))
        flipped_W = np.flip(np.flip(self.W, axis=2), axis=3)
        dLdA = np.zeros(self.A.shape)

        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                dLdA_ij = np.tensordot(dLdZ_padded[:, :, i:i+self.kernel_size, j:j+self.kernel_size], flipped_W, axes=((1, 2, 3), (0, 2, 3)))
                dLdA[:, :, i, j] = dLdA_ij

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)))

        # Call Conv2d_stride1
        conv_fwd = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(conv_fwd)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        downsampled_dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(downsampled_dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:,:,self.pad:-self.pad,self.pad:-self.pad]

        return dLdA
