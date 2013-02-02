## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

__all__ = ["Golub"]


class Golub:
    """Golub binary classifier.
    The decision velues are computed as d(t) = w(t-mu), where w 
    is defined as w_i = - (mu_i(+) - mu_i(-)) / (std_i(+) + std_i(-)) 
    and mu id defined as (mu(+) + mu(-)) / 2.
    """
    
    def __init__(self):
        """
        """
        
        self._labels = None
        self._w = None
        self._bias = None
        self._model = False

    def learn(self, x, y):
        """Learning method.
        
        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer (only two classes)
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")
        
        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k != 2:
            raise ValueError("number of classes must be = 2")
        
        idxn = yarr == self._labels[0]
        idxp = yarr == self._labels[1]
        meann = np.mean(xarr[idxn], axis=0)
        meanp = np.mean(xarr[idxp], axis=0)
        stdn = np.std(xarr[idxn], axis=0, ddof=1)
        stdp = np.std(xarr[idxp], axis=0, ddof=1)
        self._w = - ((meanp - meann) / (stdp + stdn))
        self._bias = - np.sum(self._w * (0.5 * (meanp + meann)))
        
        self._model = True

    def pred_values(self, t):
        """Returns the decision value (d(Kt)) for eache test sample.
        
        :Parameters:	
           t : 1d (one sample) or 2d array_like object
              test data ([M,] P)
        :Returns:	
           decision values : 1d (1) or 2d numpy array (M, 1)
              decision values for each observation.
        """

        if not self._model:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
        if tarr.ndim > 2:
            raise ValueError("t must be an 1d or a 2d array_like object")
        
        try:
            values = np.dot(tarr, self._w) + self._bias
        except ValueError:
            raise ValueError("t, w: shape mismatch")

        if tarr.ndim == 1:
            return np.array([values])
        else:
            return values.reshape(-1, 1)

    def pred(self, t):
        """Does classification on test vector(s) `t`.
        Returns l_1 if g(t) > 0, l_2 otherwise.
      
        :Parameters:
           t : 1d or 2d array_like object
              test sample(s) ([M,] P)
            
        :Returns:        
            p : integer or 1d numpy array
               the predicted class(es)
        """
        
        values = self.pred_values(t)

        if values.ndim == 1:
            values = values[0]
        else:
            values = np.ravel(values)

        return np.where(values > 0, self._labels[0], self._labels[1]) \
            .astype(np.int)
    
    def w(self):
        """Returns the slope coefficients.
        """
        
        if not self._model:
            raise ValueError("no model computed")
        
        return self._w

    def bias(self):
        """Returns the intercept."""
        
        if not self._model:
            raise ValueError("no model computed")

        return self._bias

    def labels(self):
        """Returns the class labels.
        """
        
        if not self._model:
            raise ValueError("no model computed")
        
        return self._labels
