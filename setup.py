from distutils.core import setup, Extension
from distutils.sysconfig import *
from distutils.util import *
import os
import os.path
import numpy

try:
   from distutils.command.build_py import build_py_2to3 \
       as build_py
except ImportError:
   from distutils.command.build_py import build_py

from Cython.Distutils import build_ext

#### data files
data_files = []
# Include gsl libs for the win32 distribution
if get_platform() == "win32":
   dlls = ["gsl.lib", "cblas.lib"]
   data_files += [("Lib\site-packages\mlpy", dlls)]
   
#### libs
if get_platform() == "win32":
   gsl_lib = ['gsl', 'cblas']
   math_lib = []
else:
   gsl_lib = ['gsl', 'gslcblas']
   math_lib = ['m']
   
#### Extra compile args
if get_platform() == "win32":
   extra_compile_args = []
else:
   extra_compile_args = ['-Wno-strict-prototypes']
   
#### Python include
py_inc = [get_python_inc()]

#### NumPy include
np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

#### scripts
scripts = []

#### cmdclass
cmdclass = {'build_py': build_py, 'build_ext': build_ext}

#### Extension modules
ext_modules = []
ext_modules += [Extension("mlpy.dtw",
                         ["mlpy/dtw/cdtw.c",
                         "mlpy/dtw/dtw.pyx"],
                         libraries=math_lib,
                         include_dirs=py_inc + np_inc)]
packages=['mlpy'],

classifiers = ['Development Status :: 5 - Production/Stable',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU General Public License (GPL)',
               'Programming Language :: C',
               'Programming Language :: C++',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: POSIX :: Linux',
               'Operating System :: POSIX :: BSD',
               'Operating System :: MacOS',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX'
               ]

setup(name='mlpy',
      version='3.5.1',
      requires=['numpy (>=1.3.0)'],
      description='mlpy - Machine Learning Py - ' \
         'High-Performance Python Package for Predictive Modeling',
      author='mlpy Developers',
      author_email='davide.albanese@gmail.com',
      maintainer='Davide Albanese',
      maintainer_email='davide.albanese@gmail.com',
      packages=packages,
      url='mlpy.sourceforge.net',
      download_url='https://sourceforge.net/projects/mlpy/',
      license='GPLv3',
      classifiers=classifiers,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      scripts=scripts,
      data_files=data_files
      )
