# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        # W shape is (out_channels, in_channels, kernel_size)
        # A[:, :, i:i+kernel_size] shape is (batch_size, in_channels, kernel_size)
        # output is (batch_size, out_channels, output_size)
        self.output_size = A.shape[2] - self.kernel_size + 1

        for i in range(self.output_size):
            kernel_out = np.tensordot(A[:, :, i:i+self.kernel_size], self.W, axes=((1, 2), (1, 2))) + self.b
            Z = np.concatenate((Z, kernel_out[:,:,np.newaxis]), axis=2) if i > 0 else kernel_out[:,:,np.newaxis]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        dLdZ_size = dLdZ.shape[-1]

        # A shape is (batch_size, in_channels, input_size)
        # dLdZ shape is (batch_size, out_channels, output_size)
        # dLdW shape is (out_channels, in_channels, kernel_size)
        for i in range(self.kernel_size):
            dLdW_i = np.tensordot(dLdZ, self.A[:, :, i:i+dLdZ_size], axes=((0, 2), (0, 2)))
            self.dLdW[:, :, i] = dLdW_i

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        padding = int(self.kernel_size - 1)
        dLdZ_padded = np.pad(dLdZ, ((0,0),(0,0),(padding,padding)))
        flipped_W = np.flip(self.W, axis=2)

        for i in range(self.A.shape[-1]):
            dLdA_i = np.tensordot(dLdZ_padded[:, :, i:i+self.kernel_size], flipped_W, axes=((1, 2), (0, 2)))
            dLdA = np.concatenate((dLdA, dLdA_i[:,:,np.newaxis]), axis=2) if i > 0 else dLdA_i[:,:,np.newaxis]

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0,0),(0,0),(self.pad,self.pad)))

        # Call Conv1d_stride1
        conv_A = self.conv1d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample1d.forward(conv_A)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        downsampled_dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA_padded = self.conv1d_stride1.backward(downsampled_dLdZ)

        # Unpad the gradient
        dLdA = dLdA_padded[:,:,self.pad:-self.pad]

        return dLdA
