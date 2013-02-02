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
from kernel_class import *


__all__ = ['LDAC', 'DLDA', 'KFDAC', 'QDA']


class LDAC:
    """Linear Discriminant Analysis Classifier.
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
        
        p = np.empty(k, dtype=np.float)
        mu = np.empty((k, xarr.shape[1]), dtype=np.float)
        cov = np.zeros((xarr.shape[1], xarr.shape[1]), dtype=np.float)

        for i in range(k):
            wi = (yarr == self._labels[i])
            p[i] = np.sum(wi) / float(xarr.shape[0])
            mu[i] = np.mean(xarr[wi], axis=0)
            xi = xarr[wi] - mu[i]
            cov += np.dot(xi.T, xi)
        cov /= float(xarr.shape[0] - k)
        covinv = np.linalg.inv(cov)
        
        self._w = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._bias = np.empty(k, dtype=np.float)

        for i in range(k):           
            self._w[i] = np.dot(covinv, mu[i])
            self._bias[i] = - 0.5 * np.dot(mu[i], self._w[i]) + \
                np.log(p[i])

        self._model = True

    def labels(self):
        """Returns the class labels.
        """

        if not self._model:
            raise ValueError("no model computed")

        return self._labels
        
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


class DLDA:
    """Diagonal Linear Discriminant Analysis classifier.
    The algorithm uses the procedure called Nearest Shrunken
    Centroids (NSC).
    """
    
    def __init__(self, delta):
        """Initialization.
        
        :Parameters:
           delta : float
              regularization parameter
        """

        self._delta = float(delta)
        self._xstd = None # s_j
        self._dprime = None # d'_kj
        self._xmprime = None # xbar'_kj
        self._p = None # class prior probability
        self._labels = None
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
        
        xm = np.mean(xarr, axis=0)
        self._xstd = np.std(xarr, axis=0, ddof=1)
        s0 = np.median(self._xstd)
        self._dprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._xmprime = np.empty((k, xarr.shape[1]), dtype=np.float)
        n = yarr.shape[0]
        self._p = np.empty(k, dtype=np.float)

        for i in range(k):
            yi = (yarr == self._labels[i])
            xim = np.mean(xarr[yi], axis=0)
            nk = np.sum(yi)
            mk = np.sqrt(nk**-1 - n**-1)
            d = (xim - xm) / (mk * (self._xstd + s0))
            
            # soft thresholding
            tmp = np.abs(d) - self._delta
            tmp[tmp<0] = 0.0
            self._dprime[i] = np.sign(d) * tmp
            
            self._xmprime[i] = xm + (mk * (self._xstd + s0) * self._dprime[i])
            self._p[i] = float(nk) / float(n)
            
        self._model = True

    def labels(self):
        """Returns the class labels.
        """
        
        return self._labels
        
    def sel(self):
        """Returns the most important features (the features that 
        have a nonzero dprime for at least one of the classes).
        """

        return np.where(np.sum(self._dprime, axis=0) != 0)[0]

    def dprime(self):
        """Return the dprime d'_kj (C, P), where C is the
        number of classes.
        """
        
        return self._dprime

    def _pred_values_one(self, t):
        values = - np.sum((t-self._xmprime)**2/self._xstd**2,
                          axis=1) + (2 * np.log(self._p))
        
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

    def _pred_probability_one(self, t):
        """Return the probability estimates"""
        
        values = - np.sum((t-self._xmprime)**2/self._xstd**2,
                          axis=1) + (2 * np.log(self._p))

        tmp = np.exp(values * 0.5)
        return tmp / np.sum(tmp)
        
    def pred_probability(self, t):
        """Returns C (number of classes) probability estimates.

        :Parameters :
           t : 1d (one sample) or 2d array_like object
              test data ([M,] P)
        :Returns :
           probability estimates : 1d (C) or 2d numpy array (M, C)
              probability estimates for each observation.
        """

        if not self._model:
            raise ValueError("no model computed")
                
        tarr = np.asarray(t, dtype=np.float)

        if tarr.ndim == 1:
            return self._pred_probability_one(tarr)
        else:
            ret = []
            for i in range(tarr.shape[0]):
                ret.append(self._pred_probability_one(tarr[i]))
            return np.array(ret)


class KFDAC:
    """Kernel Fisher Discriminant Analysis Classifier (binary).
    """
    
    def __init__(self, lmb=0.001, kernel=None):
        """Initialization.

        :Parameters:
           lmb : float (>= 0.0)
              regularization parameter
           kernel : None or mlpy.Kernel object.
              if kernel is None, K and Kt in .learn()
              and in .pred*() methods must be precomputed kernel 
              matricies, else K and Kt must be training (resp. 
              test) data in input space.
        """

        if kernel is not None:
            if not isinstance(kernel, Kernel):
                raise ValueError("kernel must be None or a mlpy.Kernel object")

        self._lmb = float(lmb)
        self._kernel = kernel
        self._labels = None
        self._alpha = None
        self._b = None
        self._x = None
        self._model = False
      
    def learn(self, K, y):
        """Learning method.

        :Parameters:
           K: 2d array_like object
              precomputed training kernel matrix (if kernel=None);
              training data in input space (if kernel is a Kernel object)
           y : 1d array_like object integer (N)
              class labels (only two classes)
        """

        Karr = np.asarray(K, dtype=np.float)
        yarr = np.asarray(y, dtype=np.int)

        if Karr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")
        
        if Karr.shape[0] != yarr.shape[0]:
            raise ValueError("K, y shape mismatch")

        if self._kernel is None:
            if Karr.shape[0] != Karr.shape[1]:
                raise ValueError("K must be a square matrix")
        else:
            self._x = Karr.copy()
            Karr = self._kernel.kernel(Karr, Karr)

        self._labels = np.unique(yarr)
        if self._labels.shape[0] != 2:
            raise ValueError("number of classes must be = 2")
        
        n = yarr.shape[0]
        
        idx1 = np.where(yarr==self._labels[0])[0]
        idx2 = np.where(yarr==self._labels[1])[0]
        n1 = idx1.shape[0]
        n2 = idx2.shape[0]
        
        K1, K2 = Karr[:, idx1], Karr[:, idx2]
        
        N1 = np.dot(np.dot(K1, np.eye(n1) - (1 / float(n1))), K1.T)
        N2 = np.dot(np.dot(K2, np.eye(n2) - (1 / float(n2))), K2.T)
        N = N1 + N2 + np.diag(np.repeat(self._lmb, n))
        Ni = np.linalg.inv(N)

        m1 = np.sum(K1, axis=1) / float(n1)
        m2 = np.sum(K2, axis=1) / float(n2)
        d = (m1 - m2)
        M = np.dot(d.reshape(-1, 1), d.reshape(1, -1))

        self._alpha = np.linalg.solve(N, d)
        self._b = - np.dot(self._alpha, (n1 * m1 + n2 * m2) / float(n))
        self._model = True
        
    def labels(self):
        """Returns the class labels.
        """

        if not self._model:
            raise ValueError("no model computed")

        return self._labels
        
    def pred_values(self, Kt):
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

        Ktarr = np.asarray(Kt, dtype=np.float)
        if self._kernel is not None:
            Ktarr = self._kernel.kernel(Ktarr, self._x)

        try:
            values = np.dot(self._alpha, Ktarr.T) + self._b
        except ValueError:
            raise ValueError("Kt, alpha: shape mismatch")

        if Ktarr.ndim == 1:
            return np.array([values])
        else:
            return values.reshape(-1, 1)

    def pred(self, Kt):
        """Does classification on test vector(s) `Kt`.
        Returns l_1 if g(Kt) > 0, l_2 otherwise.
      
        :Parameters:
           Kt : 1d or 2d array_like object
               precomputed test kernel matrix. (if kernel=None);
               test data in input space (if kernel is a Kernel object).
            
        :Returns:        
            p : integer or 1d numpy array
               the predicted class(es)
        """

        values = self.pred_values(Kt)

        if values.ndim == 1:
            values = values[0]
        else:
            values = np.ravel(values)

        return np.where(values > 0, self._labels[0], self._labels[1]) \
            .astype(np.int)
            
    def alpha(self):
        """Return alpha.
        """

        return self._alpha

    def b(self):
        """Return b.
        """

        return self._b


class QDA:
    """Quadratic Discriminant Analysis Classifier.
    """
    
    def __init__(self):
        """Initialization.
        """

        self._prior = None
        self._mean = None
        self._evecs = None
        self._evals = None
        self._labels = None
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

        n = np.min(xarr.shape)

        self._prior = np.empty(k, dtype=np.float)
        self._mean = np.empty((k, xarr.shape[1]), dtype=np.float)
        self._evecs = []
        self._evals = []

        for i in range(k):
            r = (yarr == self._labels[i])
            self._prior[i] = np.sum(r) / float(xarr.shape[0])
            self._mean[i] = np.mean(xarr[r], axis=0)
            xs = (xarr[r] - np.mean(xarr[r], axis=0)) / \
                np.sqrt(xarr[r].shape[0] - 1)
            _, s, v = np.linalg.svd(xs, full_matrices=False)
            self._evecs.append(v.T)
            self._evals.append(s**2)
        
        self._model = True

    def _pred_values_one(self, t):
        values = np.empty(self._labels.shape[0], dtype=np.float)
        
        for i in range(self._labels.shape[0]):
            tm = t - self._mean[i]   
            tmp1 = np.dot(tm, self._evecs[i])
            tmp2 = np.dot(tmp1, np.diag(1.0 / self._evals[i]))
            tmp3 = np.dot(tmp2, tmp1.T)
            values[i] = - 0.5 * np.sum(np.log(self._evals[i])) - \
                0.5 * tmp3 + np.log(self._prior[i])
        
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
        :Returns:	
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

    def labels(self):
        """Returns the class labels.
        """
        
        return self._labels
