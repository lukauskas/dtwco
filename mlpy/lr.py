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

__all__ = ['LogisticRegression']

import sys
if sys.version >= '3':
    from . import liblinear
else:
    import liblinear


class LogisticRegression:
    """Logistic Regression Classifier implementing one-vs-the rest
    multi-class strategy.
    """

    SOLVER_TYPE = {
        "l2r": "l2r_lr",
        "l2r_dual": "l2r_lr_dual",
        "l1r": "l1r"
        }
    
    def __init__(self, solver_type='l2r', C=1, eps=0.01, weight={}):
        """Initialization.

        :Parameters:
            solver_type : string
                solver, can be one of 'l2r', 'l2r_dual' (L2-regularized) 
                and 'l1r' (L1-regularized).
            C : float
                cost of constraints violation
            eps : float
                stopping criterion
            weight : dict 
                changes the penalty for some classes (if the weight for a
                class is not changed, it is set to 1). For example, to
                change penalty for classes 1 and 2 to 0.5 and 0.8
                respectively set weight={1:0.5, 2:0.8}
        """
        
        if solver_type not in self.SOLVER_TYPE:
            raise ValueError("invalid solver_type")

        self._liblinear = liblinear.LibLinear(
            solver_type=self.SOLVER_TYPE[solver_type], C=C, eps=eps,
            weight=weight)
    
    def learn(self, x, y):
        """Learning method.

        :Parameters:
           x : 2d array_like object
              training data (N, P)
           y : 1d array_like object integer
              target values (N)
        """

        self._liblinear.learn(x, y)
        
    def labels(self):
        """Returns the class labels.
        """
        
        return self._liblinear.labels()

    def w(self):
        """Returns the slope coefficients. For multiclass 
        classification returns a 2d numpy array where each
        row contains the coefficients of label i (w_i). 
        For binary classification an 1d numpy array 
        (w_1 - w_2) is returned.
        """
        
        return self._liblinear.w()

    def bias(self):
        """Returns the intercept. For multiclass 
        classification returns a 1d numpy array where each
        element contains the coefficient of label i (bias_i). 
        For binary classification a float (bias_1 - bias_2) 
        is returned.
        """

        return self._liblinear.bias()

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
        
        return self._liblinear.pred_values(t)

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

        return self._liblinear.pred(t)

    def pred_probability(self, t):
        """Returns C (number of classes) probability estimates. 

        :Parameters:
            t : 1d (one sample) or 2d array_like object
                test data ([M,] P)
            
        :Returns:
            probability estimates : 1d (C) or 2d numpy array (M, C)
                probability estimates for each sample.
        """

        return self._liblinear.pred_probability(t)
