"""
`dtwco` - implementation of constrained DTW algorithms in python.
"""
from setuptools import Extension
from setuptools import setup, find_packages
from os import path

try:
    from Cython.Build import cythonize
    import numpy
except ImportError:
    print('Ensure cython and numpy are installed before building this')
    raise

here = path.abspath(path.dirname(__file__))

extensions = cythonize([Extension("dtwco._cdtw", ["dtwco/_cdtw/cdtw.c",
                                                  "dtwco/_cdtw/dtw.pyx"],
                                  include_dirs=[numpy.get_include()])
                        ])

setup(
    name='dtwco',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.9',

    description='dtwco - implementation of constrained DTW algorithms in python',
    # TODO: long description
    long_description='dtwco - implementation of constrained DTW algorithms in python',

    # The project's main homepage.
    url='https://github.com/lukauskas/mlpy-plus-dtw',

    # Author details
    author='Saulius Lukauskas',
    author_email='saulius.lukauskas13@imperial.ac.uk',

    license='GPL-3',


    classifiers=[
        'Development Status :: 4 - Beta',

        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    keywords='dtw',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    ext_modules=extensions,

    install_requires=['numpy', 'cython'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    },
)