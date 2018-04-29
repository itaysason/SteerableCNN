import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


class SpanFilter(nn.Module):
    """
    Span Filter class, represent a cnn layer where the filters are inside the span given.
    """
    def __init__(self, in_channels, out_channels, span, stride=1, padding=0, dilation=1, bias=True):
        """
        :param span: (size_x, size_y, n) float or complex ndarry
        """

        super(SpanFilter, self).__init__()
        kernel_size = span.shape[0], span.shape[1]
        n = span.shape[2]

        # initializing the span layers
        self.real_base = nn.Conv2d(in_channels, n, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, bias=False)
        self.imag_base = nn.Conv2d(in_channels, n, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, bias=False)

        real_span = span.real
        imag_span = span.imag
        # setting the values to the span vectors
        for i in range(n):
            for j in range(in_channels):
                self.real_base.weight.data[i, j] = torch.from_numpy(real_span[:, :, i])
                self.imag_base.weight.data[i, j] = torch.from_numpy(imag_span[:, :, i])

        self.real_base.eval()
        self.imag_base.eval()

        # freezing the span layers
        for param in self.real_base.parameters():
            param.requires_grad = False

        for param in self.imag_base.parameters():
            param.requires_grad = False

        self.linear_comb = nn.Conv2d(n, out_channels, kernel_size=1, bias=False)
        self.linear_comb.weight.data[:, :, 0, 0] = torch.from_numpy(np.ones(self.linear_comb.weight.data.numpy()[:, :, 0, 0].shape, dtype='float32') * 1e-3)

    def forward(self, x):
        real_part = self.linear_comb(self.real_base(x))
        imag_part = self.linear_comb(self.imag_base(x))

        return torch.pow(real_part, 2) + torch.pow(imag_part, 2)

    def train(self, mode=True):
        self.training = True
        self.linear_comb.train()

    def eval(self):
        self.training = False
        self.linear_comb.eval()

    def view_filters(self):
        real_base = np.array(self.real_base.weight.data[:, 0, :, :])
        imag_base = np.array(self.imag_base.weight.data[:, 0, :, :])
        linear_comb = np.array(self.linear_comb.weight.data[0, :, 0, 0])

        real_filter = np.einsum('kij, k -> ij', real_base, linear_comb)
        imag_filter = np.einsum('kij, k -> ij', imag_base, linear_comb)

        plt.subplot(1, 2, 1)
        plt.imshow(np.real(real_filter), cmap='gray')
        plt.title('Real filter')

        plt.subplot(1, 2, 2)
        plt.imshow(np.real(imag_filter), cmap='gray')
        plt.title('Imaginary filter')

        plt.show()

    def parameters(self):
        return self.linear_comb.parameters()
