#include <stdlib.h>


typedef struct Path
{
  int k;
  int *px;
  int *py;
} Path;

void fill_cost_matrix_unconstrained(const double *x, const double *y, int n, int m, int n_dimensions, int squared, double *cost);
void fill_cost_matrix_with_sakoe_chiba_constraint(const double *x, const double *y, int n, int m, int n_dimensions, int squared, double *cost,
                                                  int sakoe_chiba_band_parameter);
void fill_cost_matrix_with_itakura_constraint(const double *x, const double *y, int n, int m, int n_dimensions, int squared, double *cost);
void fill_constrained_cost_matrix(const double *x, const double *y, int n, int m, int n_dimensions, int squared, double *cost,
                                  const char *constraint_matrix);

int path(double *cost, int n, int m, int startx, int starty, Path *p);
void subsequence(double *x, double *y, int n, int m, double *cost);
int subsequence_path(double *cost, int n, int m, int starty, Path *p);
