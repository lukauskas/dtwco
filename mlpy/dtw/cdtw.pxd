cdef extern from "cdtw.h":
    cdef enum MLPY_DTW_DISTANCES:
        MLPY_DTW_DISTANCE_EUCLIDEAN
        MLPY_DTW_DISTANCE_SQEUCLIDEAN
        MLPY_DTW_DISTANCE_COSINE

    ctypedef struct Path:
       int k
       int *px
       int *py

    void fill_cost_matrix_unconstrained(double *x, double *y, int n, int m, int n_dimensions, int squared, double warping_penalty,
                                        double *cost)
    void fill_cost_matrix_with_sakoe_chiba_constraint(double *x, double *y, int n, int m, int n_dimensions, int squared, double warping_penalty,
                                                      double *cost, int sakoe_chiba_band_parameter)
    void fill_cost_matrix_with_slanted_band_constraint(double *x, double *y, int n, int m, int n_dimensions, int squared, double warping_penalty,
                                                       double *cost, int width)
    void fill_cost_matrix_with_itakura_constraint(double *x, double *y, int n, int m, int n_dimensions, int squared,
                                                  double warping_penalty, double *cost)
    void fill_constrained_cost_matrix(double *x, double *y, int n, int m, int n_dimensions, int squared, double *cost, char *constraint_matrix)

    int path(double *cost, int n, int m, int startx, int starty, Path *p)
    void subsequence(double *x, double *y, int n, int m, double *cost)
    int subsequence_path(double *cost, int n, int m, int starty, Path *p)
    
