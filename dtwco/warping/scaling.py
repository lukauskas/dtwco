import numpy as np

from dtwco.warping.utils import strip_nan

def uniform_shrinking_to_length(sequence, desired_length):

    # TODO: this is from my old `dgw` package. it needs a review and documentation
    EPSILON = 1e-6

    sequence = np.asarray(sequence, dtype=float)
    sequence = strip_nan(sequence)

    current_length = len(sequence)

    if current_length == 0:
        raise ValueError('Cannot shrink sequence of length 0')
    elif current_length < desired_length:
        raise ValueError('Desired length greater than current length: {0} > {1}'.format(desired_length, current_length))
    elif current_length == desired_length:
        return sequence

    if desired_length <= 0:
        raise ValueError('Invalid length desired: {0}'.format(desired_length))

    # This is essentially how many points in the current sequence will be mapped to a single point in the newone
    shrink_factor = float(current_length) / desired_length

    try:
        ndim = sequence.shape[1]
    except IndexError:
        ndim = 0
    if ndim == 0:
        new_sequence = np.empty(desired_length)
    else:
        new_sequence = np.empty((desired_length, ndim))

    for i in range(desired_length):
        start = i * shrink_factor
        end = (i + 1) * shrink_factor

        s = 0
        d = 0

        left_bound = int(np.floor(start))

        if np.abs(start - left_bound) <= EPSILON:
            left_bound_input = 1
        else:
            left_bound_input = np.ceil(start) - start

        if left_bound_input >= EPSILON:
            s += sequence[left_bound] * left_bound_input
            d += left_bound_input

        right_bound = int(np.floor(end))
        right_bound_input = end - np.floor(end)

        if right_bound_input >= EPSILON:  # Epsilon to prevent rounding errors interfering
            s += sequence[right_bound] * right_bound_input
            d += right_bound_input

        for j in range(left_bound + 1, right_bound):
            s += sequence[j]
            d += 1.0

        assert(abs(d - shrink_factor) < 0.000001)

        new_sequence[i] = s / d

    return new_sequence