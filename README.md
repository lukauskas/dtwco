This repository is no longer maintained (nor has it been for a while). It is read only as of 2020-10-06.

[Feel free to fork it, but consider using `tslearn` instead](https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html?highlight=sakoe%20chiba#additional-constraints)

# dtwco - implementation of constrained DTW algorithms in python.

This code implements a number of constrained Dynamic Time Warping (DTW) algorithms as described in the literature.
Particularly, we implement Sakoe & Chiba constraint (Sakoe78), and Itakura Parallelogram (Itakura75).

These constraints and their implementation are inspired by DTW package in R by Toni Giorgino. [website link] http://dtw.r-forge.r-project.org/ and due credit should be given to Toni. The base of this code is also based on [`mlpy`](http://mlpy.sourceforge.net) package. If you dig through the git log, you would see that this repository used to mirror `mlpy` code and, in fact was intended to be merged to `mlpy` core at some point. Unfortunately `mlpy` project died roughly at the same time. 

Some people suggested that I re-release the constrained DTW code as a separate package, both to distance myself from the now-dead `mlpy` package, and to allow people to have a slightly more light-weight implementation of DTW to play with. As of August 2017, this is exactly what I have done, and the stripped-down version of the package is released under the same license as `mlpy`: GPL-3.0.

# Installation

This package has not been released to PyPi yet, but will be released once version hits 1.0.0.
To install it, just clone the repository, and pip install it from within, e.g.:
```
pip install -e .
```

# Examples

This is the example that was originally provided by `mlpy`, modified to use Itakura parallelogram,
which aimed to reproduce the Figure 2 of [FastDTW paper](https://dl.acm.org/citation.cfm?id=1367993). 
I have modified it to use Itakura prallelogram constraint.

```python
>>> import dtwco
>>> import matplotlib.pyplot as plt
>>> import matplotlib.cm as cm
>>> x = [0,0,0,0,1,1,2,2,3,2,1,1,0,0,0,0]
>>> y = [0,0,1,1,2,2,3,3,3,3,2,2,1,1,0,0]
>>> dist, cost, path = dtwco.dtw(x, y, constraint='itakura', dist_only=False)
>>> dist
0.0
>>> fig = plt.figure(1)
>>> ax = fig.add_subplot(111)
>>> plot1 = plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
>>> plot2 = plt.plot(path[0], path[1], 'w')
>>> xlim = ax.set_xlim((-0.5, cost.shape[0]-0.5))
>>> ylim = ax.set_ylim((-0.5, cost.shape[1]-0.5))
>>> plt.show()
```

For more information consult `dtwco.dtw` docstring.

# Moving from `mlpy.dtw` syntax

For those few of you who might have been using the older versions of this package, you can move to the new version by replacing all calls to `mlpy.dtw.dtw_std()` to `dtwco.dtw()`. If you were using `dtw_itakura`, `dtw_sakoe_chiba`, or other specified methods, these are still available, however you would see that deprecation warning is now being raised as they will be removed. Please use the top level `dtw` function instead.

# References

* (Albanese12). Davide Albanese, Roberto Visintainer, Stefano Merler, Samantha Riccadonna, Giuseppe Jurman, Cesare Furlanello (2012). mlpy: Machine Learning Python. [arxiv:arXiv:1202.6548][Albanese12]
* (Giorgino09). Toni Giorgino (2009). Computing and Visualizing Dynamic Time Warping Alignments in R: The dtw Package. Journal of Statistical Software, 31(7), 1-24, [doi:10.18637/jss.v031.i07][Giorgino09].
* (Muller07) M Muller. Information Retrieval for Music and Motion. Springer, 2007. [ISBN:3540740473][Muller07]
* (Keogh01): E J Keogh, M J Pazzani. Derivative Dynamic Time Warping. In First SIAM International Conference on Data Mining, 2001. [link][Keogh01]
* (Sakoe78) H Sakoe, & S Chiba S. Dynamic programming algorithm optimization for spoken word recognition. Acoustics, 1978 [doi:/10.1109/TASSP.1978.1163055][Sakoe78]
* (Itakura75) F Itakura. Minimum prediction residual principle applied to speech recognition. Acoustics, Speech and Signal Processing, IEEE Transactions on, 23(1), 67–72, 1975. [doi:10.1109/TASSP.1975.1162641][Itakura75] 
    
[Muller07]: https://www.springer.com/gb/book/9783540740476
[Keogh01]: https://www.ics.uci.edu/~pazzani/Publications/sdm-02.pdf
[Sakoe78]: https://doi.org/10.1109/TASSP.1978.1163055
[Itakura75]: https://dx.doi.org/10.1109/TASSP.1975.1162641
[Giorgino09]: https://dx.doi.org/10.18637/jss.v031.i07 
[Albanese12]: https://arxiv.org/abs/1202.6548
