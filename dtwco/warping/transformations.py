import numpy as np

from dtwco.warping.utils import no_nan_len


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

    nnl = no_nan_len(base_sequence)

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
