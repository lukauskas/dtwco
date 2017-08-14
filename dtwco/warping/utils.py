import numpy as np

def strip_nan(sequence):
    """
    Strips NaNs that are padded to the right of each peak if they are of unequal length
    :param sequence:
    :return:
    """
    sequence = np.asarray(sequence)

    try:
        lookup = np.all(np.isnan(sequence), axis=1)
    except ValueError:
        # Will get thrown for one-dimensional arrays
        return sequence[~np.isnan(sequence)]

    sequence = sequence[~lookup]

    if np.any(np.isnan(sequence)):
        raise ValueError('Inconsistent NaNs between dimensions')

    return sequence


def no_nan_len(sequence):
    """
    Returns length of the sequence after removing all the nans from it.
    :param sequence:
    :return:
    """
    return len(strip_nan(sequence))