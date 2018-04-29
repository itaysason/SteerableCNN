from ConverterModel.Converter import Converter
import numpy as np


def get_steerable_base(im_size, truncation, beta):
    converter = Converter(im_size, truncation, beta)
    converter.init_direct()
    ang_freqs = converter.direct_model.angular_frequency
    basis = converter.direct_model.get_samples_as_images()

    freq_to_base = []

    for freq in np.sort(np.unique(ang_freqs)):
        freq_to_base.append(basis[:, :, ang_freqs == freq].astype('complex64'))

    return freq_to_base
