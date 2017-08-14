import numpy as np


def _strip_nan(sequence):
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


def _no_nan_len(sequence):
    """
    Returns length of the sequence after removing all the nans from it.
    :param sequence:
    :return:
    """
    return len(_strip_nan(sequence))


def dtw_projection(sequence, base_sequence, dtw_path):
    """
    Projects given sequence onto a base time series using Dynamic Time Warping
    :param sequence: the sequence that will be projected onto base_sequence
    :param base_sequence: base sequence to project onto
    :param dtw_path: pre-computed DTW warping path between sequence and base sequence.
    :return: new time series of length len(base) containing sequence projected onto base
    """
    base_sequence = np.asarray(base_sequence)
    sequence = np.asarray(sequence)

    path_other, path_base = dtw_path

    current_sums = np.zeros(base_sequence.shape)
    current_counts = np.zeros(base_sequence.shape)

    nnl = _no_nan_len(base_sequence)

    try:
        filler = [np.nan] * base_sequence.shape[1]
    except IndexError:
        filler = np.nan

    for i in range(nnl, len(base_sequence)):
        current_sums[i] = filler
        current_counts[i] = filler

    for mapped_i, i in zip(path_base, path_other):
        # Go through the path and sum all points that map to the base location i together
        current_sums[mapped_i] += sequence[i]
        current_counts[mapped_i] += 1

    current_average = current_sums / current_counts

    # Append NaNs as needed
    nans_count = len(base_sequence) - len(base_sequence)
    if nans_count:
        current_average = np.concatenate((current_average,
                                          [[np.nan] * base_sequence.shape[-1]] * (nans_count)))

    return current_average
