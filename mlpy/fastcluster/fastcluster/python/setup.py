#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
  fastcluster: Fast hierarchical clustering routines for R and Python

  Copyright © 2011 Daniel Müllner
  <http://math.stanford.edu/~muellner>
'''
#import distutils.debug
#distutils.debug.DEBUG = 'yes'
from numpy.distutils.core import setup, Extension

setup(name='fastcluster', \
      version='1.1.2', \
      py_modules=['fastcluster'], \
      description='Fast hierarchical clustering routines for R and Python.', \
      long_description="""
This library provides Python functions for hierarchical clustering. It generates hierarchical clusters from distance matrices or from vector data.

Part of this module is intended to replace the functions

    linkage, single, complete, average, weighted, centroid, median, ward

in the module scipy.cluster.hierarchy with the same functionality but much faster algorithms. Moreover, the function 'linkage_vector' provides memory-efficient
clustering for vector data.

The interface is very similar to MATLAB's Statistics Toolbox API to make code easier to port from MATLAB to Python/Numpy. The core implementation of this library is in C++ for efficiency.
""",
      ext_modules=[Extension('_fastcluster',
                             ['../src/fastcluster_python.cpp'],
                  # Feel free to uncomment the line below if you use the GCC.
                  # This switches to more aggressive optimization and turns
                  # more warning switches on. No warning should appear in
                  # the compilation process.
                  #
                  # extra_compile_args=['-O3', '-Wall', '-ansi', '-Wconversion', '-Wsign-conversion'],
                  # (no -pedantic -Wextra)
     )],
      keywords=['dendrogram', 'linkage', 'cluster', 'agglomerative', 'hierarchical', 'hierarchy', 'ward'],
      author=u"Daniel Müllner",
      author_email="fastcluster@math.stanford.edu",
      license="GPLv3 <http://www.gnu.org/licenses/gpl.html>",
      classifiers = ["Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Topic :: Scientific/Engineering :: Bio-Informatics",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Programming Language :: Python",
                     "Programming Language :: C++",
                     "Operating System :: OS Independent",
                     "License :: OSI Approved :: GNU General Public License (GPL)",
                     "Intended Audience :: Science/Research",
                     "Development Status :: 5 - Production/Stable"],
      url = 'http://math.stanford.edu/~muellner',
      )
