import torch.nn as nn
from PSWF2D_utils import get_steerable_base
from SpanFilter import SpanFilter
import torch
import numpy as np


class SteerableCNN(nn.Module):
    """
    SteerableCNN class, has the same interface as regular CNN for easy use.
    Creates a CNN layer where all the filters are Steerable filters.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 truncation=10, beta=1, angular_freqs=None):

        super(SteerableCNN, self).__init__()
        basis = get_steerable_base(kernel_size, truncation, beta)

        # random choice over the angular frequencies
        if angular_freqs is None:
            angular_freqs = np.arange(len(basis))
            p = np.array([basis[i].shape[2] for i in range(len(basis))], dtype='float')
            p = np.exp(p)
            p = p / p.sum()
            angular_freqs = np.random.choice(angular_freqs, out_channels, p=p)

        angular_freq, count = np.unique(angular_freqs, return_counts=True)

        filters = []
        for i in range(len(angular_freq)):
            filters.append(SpanFilter(in_channels, int(count[i]), basis[angular_freq[i]], stride, dilation=dilation,
                                      bias=bias))

        self.pad = nn.ZeroPad2d(padding)
        self.filters = nn.ModuleList(filters)

    def forward(self, x):
        x = self.pad(x)
        tmp = []
        for i in range(len(self.filters)):
            tmp.append(self.filters[i](x))
        x = torch.cat(tmp, dim=1)
        return x

    def out_channels(self):
        return sum(self.angular_frequencies())

    def angular_frequencies(self):
        return [self.filters[i].linear_comb.out_channels for i in range(len(self.filters))]
