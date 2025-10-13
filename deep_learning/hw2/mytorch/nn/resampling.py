import numpy as np


class Upsample1d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        output_width = self.upsampling_factor * (A.shape[2] - 1) + 1
        padded_matrix = np.zeros((A.shape[0], A.shape[1], output_width))
        padded_matrix[:, :, ::self.upsampling_factor] = A

        return padded_matrix

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        return dLdZ[:, :, ::self.upsampling_factor]


class Downsample1d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A = A
        Z = A[:, :, ::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        padded_matrix = np.zeros_like(self.A)
        padded_matrix[:, :, ::self.downsampling_factor] = dLdZ

        return padded_matrix


class Upsample2d():
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        output_height = self.upsampling_factor * (A.shape[2] - 1) + 1
        output_width = self.upsampling_factor * (A.shape[3] - 1) + 1
        padded_matrix = np.zeros((A.shape[0], A.shape[1], output_height, output_width))
        padded_matrix[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return padded_matrix

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        return dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]


class Downsample2d():
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        self.A = A
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        padded_matrix = np.zeros_like(self.A)
        padded_matrix[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return padded_matrix
