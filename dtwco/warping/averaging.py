import numpy as np

from dtwco.warping.scaling import uniform_shrinking_to_length
from dtwco.warping.utils import no_nan_len

def sdtw_averaging(sequence_a, sequence_b, dtw_path,
                   weight_a=1.0, weight_b=1.0,
                   shrink=True):
    """
    Implements Scaled Dynamic Time Warping Path Averaging as described in [#Niennattrakul:2009ep]
    .. [#Niennattrakul:2009ep] Vit Niennattrakul and Chotirat Ann Ratanamahatana "Shape averaging under Time Warping",
       2009 6th International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)
    :param sequence_a: sequence A
    :param sequence_b: sequence B
    :param dtw_path: computed mapped path between the sequences.
    :param weight_a: weight of sequence A
    :param weight_b: weight of sequence B
    :param shrink: if set to true the data will be shrunk to the length of maximum seq
    :return:
    """
    sequence_a = np.asarray(sequence_a, dtype=float)
    sequence_b = np.asarray(sequence_b, dtype=float)

    dtw_path = zip(dtw_path[0], dtw_path[1])  # Rezip this for easier traversal

    averaged_path = []

    prev = None

    # The paper does not explicitly say how to round this
    diagonal_coefficient = int((weight_a + weight_b) / 2.0)
    for a, b in dtw_path:

        item = (weight_a * sequence_a[a] + weight_b * sequence_b[b]) / (weight_a + weight_b)
        if prev is None:
            extension_coefficient = diagonal_coefficient
        else:
            if prev[0] == a:  # The path moved from (i,j-1) to (i,j)
                # assert(prev[1] + 1 == b)
                extension_coefficient = weight_a
            elif prev[1] == b:  # The path moved from (i-1,j) to (i,j)
                # assert(prev[0] + 1 == a)
                extension_coefficient = weight_b
            else:  # Path moved diagonally from (i-1,j-1) to (i,j)
                # assert(prev[0] + 1 == a)
                # assert(prev[1] + 1 == b)
                extension_coefficient = diagonal_coefficient

        new_items = [item] * extension_coefficient
        averaged_path.extend(new_items)
        prev = (a, b)

    averaged_path = np.asarray(averaged_path, dtype=float)
    if shrink:
        averaged_path = uniform_shrinking_to_length(averaged_path, max(no_nan_len(sequence_a),
                                                                       no_nan_len(sequence_b)))
    return averaged_path