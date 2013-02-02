## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2012 mlpy Developers.

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

__all__ = ['GaussianNB']


class GaussianNB:
    """Gaussian Naive Bayes' Classifier.
    """
    
    def __init__(self):
        """Initialization.
        """

        self._labels = None
        self._mean = None
        self._std = None
        self._prior = None
        self._model = False
      
    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """
        
        xarr = np.asarray(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]

        if k < 2:
            raise ValueError("number of classes must be >= 2")     
        
        self._prior = np.empty(k, dtype=np.float)
        self._mean = np.empty((k, xarr.shape[1]), dtype=np.float)
        
        for i in range(k):
            r = (yarr == self._labels[i])
            self._prior[i] = np.sum(r) / float(xarr.shape[0])
            self._mean[i] = np.mean(xarr[r], axis=0)
        
        self._std = np.std(xarr, axis=0)

        self._model = True

    def labels(self):
        """Returns the class labels.
        """

        if not self._model:
            raise ValueError("no model computed")

        return self._labels

    def _pred_values_one(self, t):
        values = np.empty(self._labels.shape[0], dtype=np.float)
        
        for i in range(self._labels.shape[0]):
            values[i] = - 0.5 *np.sum(((t - self._mean[i]) / self._std)**2) \
                +  np.log(self._prior[i])
            
        if self._labels.shape[0] == 2:
            return np.array([values[0] - values[1]])
        else:
            return values

    def _pred_one(self, t):
        values = self._pred_values_one(t)
        
        if self._labels.shape[0] == 2:
            if values > 0:
                return self._labels[0]
            else:
                return self._labels[1]
        else:
            return self._labels[np.argmax(values)]

    def pred_values(self, t):
        """Returns D decision values for eache test sample. 
        D is 1 if there are two classes (d(t) = d_1(t) - d_2(t)) 
        and it is the number of classes (d_1(t), d_2(t), ..., d_C(t)) 
        otherwise.
        
        :Parameters :	
           t : 1d (one sample) or 2d array_like object
              test data ([M,] P)
        :Returns :	
           decision values : 1d (D) or 2d numpy array (M, D)
              decision values for each observation.
        """

        if not self._model:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
                
        if tarr.ndim == 1:
            return self._pred_values_one(tarr)
        else:
            values = []
            for i in range(tarr.shape[0]):
                values.append(self._pred_values_one(tarr[i]))
            return np.array(values)

    def pred(self, t):
        """Does classification on test vector(s) `t`.
        Returns the class with the highest decision value.
        
        :Parameters:
            t : 1d (one sample) or 2d array_like object
               test sample(s) ([M,] P)
            
        :Returns:        
            p : integer or 1d numpy array
               predicted class(es)
        """
                                
        if not self._model:
            raise ValueError("no model computed")

        tarr = np.asarray(t, dtype=np.float)
                
        if tarr.ndim == 1:
            return self._pred_one(tarr)
        else:
            pred = []
            for i in range(tarr.shape[0]):
                pred.append(self._pred_one(tarr[i]))
            return np.array(pred)
