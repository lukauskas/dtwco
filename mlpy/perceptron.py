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

__all__ = ["Perceptron"]


def perceptron(x, y, alpha, thr, maxiters, w, bias):
    
    labels = np.unique(y)
    
    # binarize y
    r = []
    for l in labels:
        r.append((y == l).astype(np.int))
    r = np.array(r).T

    for i in range(1, maxiters+1):
        v = np.dot(x, w.T) + bias
        
        yp = labels[np.argmax(v, axis=1)]
        err = np.sum(y != yp) / float(y.shape[0])
        if err <= thr:
            return i-1, err

        rn = np.where(v > 0, 1, 0)
       
        for k in range(r.shape[1]):
            d = r[:, k] - rn[:, k]
            w[k] += alpha * np.dot(x.T, d)
            bias[k] += alpha * np.sum(d)
        
    v = np.dot(x, w.T) + bias
    yp = labels[np.argmax(v, axis=1)]
    err = np.sum(y != yp) / float(y.shape[0])
    return i, err


class Perceptron:
    """Offline Single Layer Perceptron Classifier, trained
    by stochastic gradient descent.
    """

    def __init__(self, alpha=0.1, thr=0.0, maxiters=1000):
        """The algorithm stops when the iteration error is less
        or equal than `thr`, or a predetermined number of 
        iterations (`maxiters`) have been completed.

        :Parameters:
           alpha : float, in range (0.0, 1]
              learning rate
           thr : float, in range [0.0, 1.0]
              iteration error (e.g. thr=0.10 for error=10%) 
           maxiters : integer (>0)
              maximum number of iterations
        """

        self._alpha = alpha # learning rate, where 0.0 < alpha <= 1
        self._thr = float(thr) # error threshold
        self._maxiters = maxiters
              
        self._labels = None
        self._w = None # slope
        self._bias = None # intercept
        self._err = None
        self._iters = None
        self._model = False

    def learn(self, x, y):
        """Learning method.
        
        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """

        xarr = np.array(x, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)
        
        if xarr.ndim != 2:
            raise ValueError("x must be a 2d array_like object")
        
        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if xarr.shape[0] != yarr.shape[0]:
            raise ValueError("x, y: shape mismatch")

        self._labels = np.unique(yarr)
        k = self._labels.shape[0]
        
        if k < 2:
            raise ValueError("number of classes must be >= 2")
        
        self._w = np.zeros((k, x.shape[1]), dtype=np.float)
        self._bias = np.zeros((k, ), dtype=np.float)

        self._iters, self._err = perceptron(xarr, yarr, self._alpha, self._thr, 
            self._maxiters, self._w, self._bias)

        self._model = True
    
    def _pred_values_one(self, t):
        values = np.empty(self._labels.shape[0], dtype=np.float)
        
        for i in range(self._labels.shape[0]):
            values[i] = np.dot(t, self._w[i]) + self._bias[i]
            
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
        
        :Parameters:
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
    
    def w(self):
        """Returns the slope coefficients. For multiclass 
        classification returns a 2d numpy array where each
        row contains the coefficients of label i (w_i). 
        For binary classification an 1d numpy array 
        (w_1 - w_2) is returned.
        """
        
        if not self._model:
            raise ValueError("no model computed")

        if self._labels.shape[0] == 2:
            return self._w[0] - self._w[1]
        else:
            return self._w

    def labels(self):
        """Returns the class labels.
        """
        
        if not self._model:
            raise ValueError("no model computed")

        return self._labels

    def bias(self):
        """Returns the intercept. For multiclass 
        classification returns a 1d numpy array where each
        element contains the coefficient of label i (bias_i). 
        For binary classification a float (bias_1 - bias_2) 
        is returned.
        """
        
        if not self._model:
            raise ValueError("no model computed")

        if self._labels.shape[0] == 2:
            return self._bias[0] - self._bias[1]
        else:
            return self._bias

    def err(self):
        """Returns the iteration error"""
                
        if not self._model:
            raise ValueError("no model computed.")

        return self._err
    
    def iters(self):
        """Returns the number of iterations performed"""
        
        if not self._model:
            raise ValueError("no model computed.")

        return self._iters
