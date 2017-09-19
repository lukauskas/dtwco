# -*- coding: utf-8 -*-
from dtwco._c import dtw as _c

def dtw(x, y, dist_only=True, metric='euclidean', constraint=None, k=None, warping_penalty=0):
    """Standard DTW as described in [Muller07]_,
        using the Euclidean distance (absolute value
        of the difference) or squared Euclidean distance
        (as in [Keogh01]_) as local cost measure.

        :Parameters:
           x : 1d array_like object (N)
              first sequence
           y : 1d array_like object (M)
              second sequence
           dist_only : bool
              compute only the distance
           metric : 'euclidean', 'sqeuclidean' or 'cosine'
              distance metric to use
           constraint: string
              one of the following:
                 None or ('None') : unconstrained DTW.
                 'sakoe_chiba': DTW constrained by Sakoe & Chiba band of width 2k + 1 (requires value of k set), see [Sakoe78]
                 'slanted_band': Generalisation of Sakoe & Chiba constraint that supports sequences of different lengths
                 'itakura'    : DTW constrained by Itakura Parallelogram, see
           k : int
              parameter required by sakoe_chiba and slanted_band constraints.
           warping_penalty: double
              warping penalty to impose on non-diagonal path changes (default: 0)
           :Returns:
           dist : float
              unnormalized minimum-distance warp path
              between sequences
           cost : 2d numpy array (N,M) [if dist_only=False]
              accumulated cost matrix
           path : tuple of two 1d numpy array (path_x, path_y) [if dist_only=False]
              warp path

        Specifics of constraints::
            'sakoe_chiba': warping path is constrained along the diagonal such that for
                           every i, j, the following constraint is true: abs(i-j) <= k,
                           see [Sakoe78]
            'slanted_band': generalisation of Sakoe & Chiba Band. Warping path is constrained by
                            the following equation abs(i*len(x)/len(k)-j) <= k . See implementation
                            of R package dtw.
            'itakura': The warping path is constrained by Itakura Parallelogram. See [Itakura75].


        .. [Muller07] M Muller. Information Retrieval for Music and Motion. Springer, 2007.
        .. [Keogh01] E J Keogh, M J Pazzani. Derivative Dynamic Time Warping. In First SIAM International Conference on Data Mining, 2001.
        .. [Sakoe78] H Sakoe, & S Chiba S. Dynamic programming algorithm optimization for spoken word recognition. Acoustics, 1978
        .. [Itakura75] F Itakura. Minimum prediction residual principle applied to speech recognition. Acoustics, Speech and Signal Processing, IEEE Transactions on, 23(1), 67–72, 1975. doi:10.1109/TASSP.1975.1162641.
        """

    return _c.dtw_std(x, y,
                      dist_only=dist_only,
                      metric=metric,
                      constraint=constraint,
                      k=k,
                      warping_penalty=warping_penalty)

def dtw_subsequence(x, y):
    """Subsequence DTW as described in [Muller07]_,
        assuming that the length of `y` is much larger
        than the length of `x` and using the Manhattan
        distance (absolute value of the difference) as
        local cost measure.

        Returns the subsequence of `y` that are close to `x`
        with respect to the minimum DTW distance.

        :Parameters:
           x : 1d array_like object (N)
              first sequence
           y : 1d array_like object (M)
              second sequence

        :Returns:
           dist : float
              unnormalized minimum-distance warp path
              between x and the subsequence of y
           cost : 2d numpy array (N,M) [if dist_only=False]
              complete accumulated cost matrix
           path : tuple of two 1d numpy array (path_x, path_y)
              warp path

        """

    return _c.dtw_subsequence(x, y)
