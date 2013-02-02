#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TBD test single on integer matrices for hamming/jaccard

print u'''Test program for the 'fastcluster' package.

Copyright © 2011 Daniel Müllner, <http://math.stanford.edu/~muellner>

If everything is OK, the test program will run forever, without an error
message.
'''.encode('utf-8')
import fastcluster as fc
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import sys


import atexit
def print_seed():
  print("Seed: {0}".format(seed))
atexit.register(print_seed)

seed  = np.random.randint(0,1e9)

print_seed()
np.random.seed(seed)
abstol = 1e-14 # absolute tolerance
rtol = 1e-13 # relative tolerance

def test_all(n,dim):
  method = 'single'

  # metrics for boolean vectors
  pcd = np.array(np.random.random_integers(0,1,(n,dim)), dtype=np.bool)
  pcd2 = pcd.copy()
  for metric in ('hamming', 'jaccard', 'yule', 'matching', 'dice', #'kulsinski',
                 'rogerstanimoto',
                 #'sokalmichener',
                 # exclude, bug in older Scipy versions
                 # http://projects.scipy.org/scipy/ticket/1486
                 'russellrao', 'sokalsneath',
                 #'kulsinski'
                 # exclude, bug in older Scipy versions
                 # http://projects.scipy.org/scipy/ticket/1484
                 ):
    sys.stdout.write("Metric: " + metric + "...")
    D = pdist(pcd, metric)
    Z2 = fc.linkage_vector(pcd, method, metric)
    if np.any(pcd2!=pcd):
      raise AssertionError('Input array was corrupted.', pcd)
    test(Z2, method, D)

  # metrics for real vectors
  bound = math.sqrt(n)
  pcd = np.random.random_integers(-bound,bound,(n,dim))
  for metric in ['euclidean', 'sqeuclidean', 'cityblock', 'chebychev', 'minkowski',
                 'cosine', 'correlation', 'hamming', 'jaccard',
                 #'canberra',
                 # exclude, bug in older Scipy versions
                 # http://projects.scipy.org/scipy/ticket/1430
                 'braycurtis', 'seuclidean', 'mahalanobis',
                 'user']:
    sys.stdout.write("Metric: " + metric + "...")
    if metric=='minkowski':
      p = np.random.uniform(1.,10.)
      sys.stdout.write("p: " + str(p) + "...")
      D = pdist(pcd, metric, p)
      Z2 = fc.linkage_vector(pcd, method, metric, p)
    elif metric=='user':
      # Euclidean metric as a user function
      fn = (lambda u, v: np.sqrt(((u-v)*(u-v).T).sum()))
      D = pdist(pcd, fn)
      Z2 = fc.linkage_vector(pcd, method, fn)
    else:
      D = pdist(pcd, metric)
      Z2 = fc.linkage_vector(pcd, method, metric)
    test(Z2, method, D)

  #print pcd
  D = pdist(pcd)
  for method in ['ward', 'centroid', 'median']:
    Z2 = fc.linkage_vector(pcd, method)
    test(Z2, method, D)

def test(Z2, method, D):
    #print(np.diff(Z2[:,2]))
    sys.stdout.write("Method: " + method + "...")
    I = np.array(Z2[:,:2], dtype=int)

    Ds = squareform(D)
    n = Ds.shape[0]
    row_repr = np.arange(2*n-1)
    row_repr[n:] = -1
    size = np.ones(n, dtype=np.int)

    Ds.flat[::n+1] = np.inf

    mins = np.empty(n-1)

    for i in xrange(n-1):
      for j in xrange(n-1):
        mins[j] = np.min(Ds[j,j+1:])
      gmin = np.min(mins)
      if abs(Z2[i,2]-gmin)/max(abs(Z2[i,2]),abs(gmin)) > rtol and \
            abs(Z2[i,2]-gmin)>abstol:
          raise AssertionError('Not the global minimum in step {2}: {0}, {1}'.format(Z2[i,2], gmin,i), squareform(D))
      i1, i2 = np.take(row_repr, I[i,:])
      if (i1<0):
        raise AssertionError('Negative index i1.', squareform(D))
      if (i2<0):
        raise AssertionError('Negative index i2.', squareform(D))
      if I[i,0]>=I[i,1]:
        raise AssertionError('Convention violated.', squareform(D))
      if i1>i2:
        i1, i2 = i2, i1
      if abs(Ds[i1,i2]-gmin)/max(abs(Ds[i1,i2]),abs(gmin)) > rtol and \
            abs(Ds[i1,i2]-gmin)>abstol:
          raise AssertionError('The global minimum is not at the right place in step {5}: ({0}, {1}): {2} != {3}. Difference: {4}'
                               .format(i1, i2, Ds[i1, i2], gmin, Ds[i1, i2]-gmin, i), squareform(D))

      s1 = size[i1]
      s2 = size[i2]
      S = s1+s2
      if method=='single':
          Ds[:i1,i2]   = np.min( Ds[:i1,(i1,i2)],axis=1)
          Ds[i1:i2,i2] = np.minimum(Ds[i1,i1:i2],Ds[i1:i2,i2])
          Ds[i2,i2:]   = np.min( Ds[(i1,i2),i2:],axis=0)
      elif method=='complete':
          Ds[:i1,i2]   = np.max( Ds[:i1,(i1,i2)],axis=1)
          Ds[i1:i2,i2] = np.maximum(Ds[i1,i1:i2],Ds[i1:i2,i2])
          Ds[i2,i2:]   = np.max( Ds[(i1,i2),i2:],axis=0)
      elif method=='average':
          Ds[:i1,i2]   = ( Ds[:i1,i1]*s1 + Ds[:i1,i2]*s2 ) / S
          Ds[i1:i2,i2] = ( Ds[i1,i1:i2]*s1 + Ds[i1:i2,i2]*s2 ) / S
          Ds[i2,i2:]   = ( Ds[i1,i2:]*s1 + Ds[i2,i2:]*s2 ) / S
      elif method=='weighted':
          Ds[:i1,i2]   = np.mean( Ds[:i1,(i1,i2)],axis=1)
          Ds[i1:i2,i2] = ( Ds[i1,i1:i2] + Ds[i1:i2,i2] ) / 2
          Ds[i2,i2:]   = np.mean( Ds[(i1,i2),i2:],axis=0)
      elif method=='ward':
          Ds[:i1,i2]   = np.sqrt((np.square(Ds[:i1,i1])*(s1+size[:i1])
                         -gmin*gmin*size[:i1]
                         +np.square(Ds[:i1,i2])*(s2+size[:i1]))/(S+size[:i1]))
          Ds[i1:i2,i2] = np.sqrt((np.square(Ds[i1,i1:i2])*(s1+size[i1:i2])
                         -gmin*gmin*size[i1:i2]
                         +np.square(Ds[i1:i2,i2])*(s2+size[i1:i2]))/(S+size[i1:i2]))
          Ds[i2,i2:]   = np.sqrt((np.square(Ds[i1,i2:])*(s1+size[i2:])
                         -gmin*gmin*size[i2:]
                         +np.square(Ds[i2,i2:])*(s2+size[i2:]))/(S+size[i2:]))
      elif method=='centroid':
          Ds[:i1,i2]   = np.sqrt((np.square(Ds[:i1,i1])*s1
                         +np.square(Ds[:i1,i2])*s2)*S-gmin*gmin*s1*s2)/S
          Ds[i1:i2,i2] = np.sqrt((np.square(Ds[i1,i1:i2])*s1
                         +np.square(Ds[i1:i2,i2])*s2)*S-gmin*gmin*s1*s2)/S
          Ds[i2,i2:]   = np.sqrt((np.square(Ds[i1,i2:])*s1
                         +np.square(Ds[i2,i2:])*s2)*S-gmin*gmin*s1*s2)/S
      elif method=='median':
          Ds[:i1,i2]   = np.sqrt((np.square(Ds[:i1,i1])+np.square(Ds[:i1,i2]))*2
                         -gmin*gmin)/2
          Ds[i1:i2,i2] = np.sqrt((np.square(Ds[i1,i1:i2])+np.square(Ds[i1:i2,i2]))*2
                         -gmin*gmin)/2
          Ds[i2,i2:]   = np.sqrt((np.square(Ds[i1,i2:])+np.square(Ds[i2,i2:]))*2
                         -gmin*gmin)/2
      else:
          raise ValueError('Unknown method.')

      Ds[i1, i1:n] = np.inf
      Ds[:i1, i1] = np.inf
      row_repr[n+i] = i2
      size[i2] = S
    print "OK."

while True:
  dim = np.random.random_integers(2,12)
  n = np.random.random_integers(max(2*dim,5),200)

  print 'Dimension: {0}'.format(dim)
  print 'Number of points: {0}'.format(n)

  try:
    test_all(n,dim)
  except AssertionError as E:
    print E[0]
    print E[1]
    sys.exit()
