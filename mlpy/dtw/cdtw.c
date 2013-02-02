/*  
    This code is written by Davide Albanese <davide.albanese@gmail.com>.
    (C) mlpy Developers.

    This program is free software: you can redistribute it and/or modify
    it underthe terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cdtw.h"

double
min3(double a, double b, double c)
{
  double min;
  
  min = a;
  if (b < min)
    min = b;
  if (c < min)
    min = c;
  return min;
}

int max2(int a, int b)
{
    if (a >= b) return a;
    else return b;
}

int min2(int a, int b)
{
    if (a <= b) return a;
    else return b;
}


// Paliwal adjustment window used for restricting the warping function
// r: window length
int
paliwal_window(int i, int j, int n, int m, int r)
{
  double s, f;
  
  s = ((double) m) / n;
  f = fabs(i - (((double) j) / s));
  
  if (f <= r)
    return 1;
  else
    return 0;
}

// euclidean distance
double e_dist(double x, double y)
{
  return fabs(x - y);
}

// squared euclidean distance
double se_dist(double x, double y)
{
  return pow(x - y, 2);
}

// Distance function selector
double (*distance_function(int squared))(double, double) {
    if (squared == 0)
        return &e_dist;
    else
        return &se_dist;
}

// Fills the cost matrix, *cost, without any constraints - O(nm)
void
fill_cost_matrix_unconstrained(const double *x, const double *y, int n, int m, int squared, double *cost)
{
     double (*dist)(double, double);
     dist = distance_function(squared);

     int i, j;
     cost[0] = (*dist)(x[0], y[0]);

      for (i=1; i<n; i++)
          cost[i*m] = (*dist)(x[i], y[0]) + cost[(i-1)*m];

      for (j=1; j<m; j++)
          cost[j] = (*dist)(x[0], y[j]) + cost[(j-1)];

      for (i=1; i<n; i++)
        for (j=1; j<m; j++)
             cost[i*m+j] = (*dist)(x[i], y[j]) +
    	        min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);
}

// Fills the cost matrix, *cost, with respect to Sakoe & Chiba band constraint |i-j| < ks  -- O(k*n)
void
fill_cost_matrix_with_sakoe_chiba_constraint(const double *x, const double *y, int n, int m, int squared, double *cost,
                                             int sakoe_chiba_band_parameter)
{
      double (*dist)(double, double);
      dist = distance_function(squared);

      int i, j;

      // Fill cost matrix with infinities first
      for (i = 0; i < n*m; i++)
          cost[i] = INFINITY;

      // Initialise
      cost[0] = (*dist)(x[0], y[0]);

      for (i=1; i<min2(n, sakoe_chiba_band_parameter+1); i++)
          cost[i*m] = (*dist)(x[i], y[0]) + cost[(i-1)*m];
      for (j=1; j<min2(m, sakoe_chiba_band_parameter+1); j++)
          cost[j] = (*dist)(x[0], y[j]) + cost[(j-1)];

      // Fill only the columns that satisfy |i-j| <= sakoe_chiba_band_parameter
      for (i=1; i<n; i++)
        for (j=max2(i-sakoe_chiba_band_parameter, 1); j<min2(m, i+sakoe_chiba_band_parameter+1); j++)
             cost[i*m+j] = (*dist)(x[i], y[j]) +
    	        min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);
}


// Implements itakura constraint. This is largely based on the following code snippet from R's dtw module
//ok<- 	(jw <  2*iw) &
// 		(iw <= 2*jw) &
//		(iw >= n-1-2*(m-jw)) &
//		(jw >  m-1-2*(n-iw)) ;
int itakura_constraint(int i, int j, int n, int m) {
    return (j < 2*i) & (i <= 2*j) & (i>= n-1-2*(m-j)) & (j > m-1-2*(n-i));
}

// Fill cost matrix constrained by Itakura Paralellogram
void
fill_cost_matrix_with_itakura_constraint(const double *x, const double *y, int n, int m, int squared, double *cost)
{
    double (*dist)(double, double);
    dist = distance_function(squared);

    int i, j;

    // Fill cost matrix with infinities first
    for (i = 0; i < n*m; i++)
      cost[i] = INFINITY;

    // Initialise
    cost[0] = (*dist)(x[0], y[0]);

    for (i=1; i<n; i++)
    {
        if (!itakura_constraint(i, 0, n, m)) continue;
        cost[i*m] = (*dist)(x[i], y[0]) + cost[(i-1)*m];
    }
    for (j=1; j<m; j++)
    {
        if (!itakura_constraint(0,j,n,m)) continue;
        cost[j] = (*dist)(x[0], y[j]) + cost[(j-1)];
    }
    for (i=1; i<n; i++)
    {
        for (j=1; j<m; j++)
        {
             if (!itakura_constraint(i,j,n,m))
                 continue;

             cost[i*m+j] = (*dist)(x[i], y[j]) +
                min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);
        }
    }
}

// Fills arbitrarily constrained matrix
// The constraint is specified by boolean array *constraint_matrix
void
fill_constrained_cost_matrix(const double *x, const double *y, int n, int m, int squared, double *cost,
                             const char *constraint_matrix)
{
    double (*dist)(double, double);
    dist = distance_function(squared);

    int i, j;

    // Fill cost matrix with infinities first
    for (i = 0; i < n*m; i++)
      cost[i] = INFINITY;

    // Initialise
    cost[0] = (*dist)(x[0], y[0]);

    for (i=1; i<n; i++)
    {
        if (!constraint_matrix[i*m]) continue;
        cost[i*m] = (*dist)(x[i], y[0]) + cost[(i-1)*m];
    }
    for (j=1; j<m; j++)
    {
        if (!constraint_matrix[j]) continue;
        cost[j] = (*dist)(x[0], y[j]) + cost[(j-1)];
    }
    for (i=1; i<n; i++)
    {
        for (j=1; j<m; j++)
        {
             if (!constraint_matrix[i*m+j])
                 continue;

             cost[i*m+j] = (*dist)(x[i], y[j]) +
                min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);
        }
    }
}

// Compute the warp path starting at cost[startx, starty]
// If startx = -1 -> startx = n-1; if starty = -1 -> starty = m-1
int
path(double *cost, int n, int m, int startx, int starty, Path *p)
{
  int i, j, k, z1, z2;
  int *px;
  int *py;
  double min_cost;
  
  if ((startx >= n) || (starty >= m))
    return 0;
  
  if (startx < 0)
    startx = n - 1;
  
  if (starty < 0)
    starty = m - 1;
      
  i = startx;
  j = starty;
  k = 1;
  
  // allocate path for the worst case
  px = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
  py = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
  
  px[0] = i;
  py[0] = j;
  
  while ((i > 0) || (j > 0))
    {
      if (i == 0)
	j--;
      else if (j == 0)
	i--;
      else
	{
	  min_cost = min3(cost[(i-1)*m+j],
			  cost[(i-1)*m+(j-1)], 
			  cost[i*m+(j-1)]);
	  
	  if (cost[(i-1)*m+(j-1)] == min_cost)
	    {
	      i--;
	      j--;
	    }
	  else if (cost[i*m+(j-1)] == min_cost)
	    j--;
	  else
	    i--;
	}
      
      px[k] = i;
      py[k] = j;
      k++;      
    }
  
  p->px = (int *) malloc (k * sizeof(int));
  p->py = (int *) malloc (k * sizeof(int));
  for (z1=0, z2=k-1; z1<k; z1++, z2--)
    {
      p->px[z1] = px[z2];
      p->py[z1] = py[z2];
    }
  p->k = k;
  
  free(px);
  free(py);
  
  return 1;
}


//
void
subsequence(double *x, double *y, int n, int m, double *cost)
{
  int i, j;
    
  cost[0] = fabs(x[0]-y[0]);
  
  for (i=1; i<n; i++)
    cost[i*m] = fabs(x[i]-y[0]) + cost[(i-1)*m];
  
  for (j=1; j<m; j++)
    cost[j] = fabs(x[0]-y[j]); // subsequence variation: D(0,j) := c(x0, yj)
  
  for (i=1; i<n; i++)
    for (j=1; j<m; j++)
      cost[i*m+j] = fabs(x[i]-y[j]) +
	min3(cost[(i-1)*m+j], cost[(i-1)*m+(j-1)], cost[i*m+(j-1)]);

}

  
int 
subsequence_path(double *cost, int n, int m, int starty, Path *p)
{
  int i, z1, z2;
  int a_star;
  int *tmpx, *tmpy;

  // find path
  if (!path(cost, n, m, -1, starty, p))
    return 0;
  
  // find a_star
  a_star = 0;
  for (i=1; i<p->k; i++)
    if (p->px[i] == 0)
      a_star++;
    else
      break;
  
  // rebuild path
  tmpx = p->px;
  tmpy = p->py;
  p->px = (int *) malloc ((p->k-a_star) * sizeof(int));
  p->py = (int *) malloc ((p->k-a_star) * sizeof(int));
  for (z1=0, z2=a_star; z2<p->k; z1++, z2++)
    {
      p->px[z1] = tmpx[z2];
      p->py[z1] = tmpy[z2];
    }
  p->k = p->k-a_star;
  
  free(tmpx);
  free(tmpy);
  
  return 1;
}
