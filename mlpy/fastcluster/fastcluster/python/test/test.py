#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

seed = np.random.randint(0,1e9)

np.random.seed(seed)
#abstol = 1e-14 # absolute tolerance
rtol = 1e-14 # relative tolerance

def test_all():
  D2 = D.copy()
  for method in ['single', 'complete', 'average', 'weighted', 'ward', 'centroid', 'median']:
    Z2 = fc.linkage(D, method)
    if np.any(D2!=D):
      raise AssertionError('Input array was corrupted.')
    test(Z2, method)

def test(Z2, method):
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
      if (Z2[i,2]-gmin)/max(abs(Z2[i,2]),abs(gmin)) > rtol:
          raise AssertionError('Not the global minimum in step {2}: {0}, {1}'.format(Z2[i,2], gmin, i))
      i1, i2 = np.take(row_repr, I[i,:])
      if (i1<0):
        raise AssertionError('Negative index i1.')
      if (i2<0):
        raise AssertionError('Negative index i2.')
      if I[i,0]>=I[i,1]:
        raise AssertionError('Convention violated.')
      if i1>i2:
        i1, i2 = i2, i1
      if (Ds[i1,i2]-gmin)/max(abs(Ds[i1,i2]),abs(gmin)) > rtol:
          raise AssertionError('The global minimum is not at the right place: ({0}, {1}): {2} != {3}. Difference: {4}'
                               .format(i1, i2, Ds[i1, i2], gmin, Ds[i1, i2]-gmin))

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
  dim = np.random.random_integers(2,20)
  n = np.random.random_integers(5,100)

  print 'Dimension: {0}'.format(dim)
  print 'Number of points: {0}'.format(n)
  pcd = np.random.randn(n,dim)
  D = pdist(pcd)

  try:
    print "Real distance values:"
    test_all()
    D = np.round(D*n/4)
    print "Integer distance values:"
    test_all()
  except AssertionError as E:
    print E
    print squareform(D)
    sys.exit()
