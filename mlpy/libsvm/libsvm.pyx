## This code is written by Davide Albanese, <albanese@fbk.eu>
## (C) 2010 mlpy Developers.

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
cimport numpy as np
from libc.stdlib cimport *

from clibsvm cimport *
cimport cython

from kernel_class import *
   
cdef void print_null(char *s):
   pass


# array 1D to svm node
@cython.boundscheck(False)
cdef svm_node *array1d_to_node(np.ndarray[np.float_t, ndim=1] x, serial=None):
    cdef int i
    cdef svm_node *ret

    ret = <svm_node*> malloc ((x.shape[0]+2) * sizeof(svm_node))
            
    ret[0].index = 0
    if serial is not None:
        ret[0].value = serial
    for i in range(1, x.shape[0]+1):
        ret[i].index = i
        ret[i].value = x[i-1]
    ret[x.shape[0]+1].index = -1

    return ret

# array 2D to svm node
@cython.boundscheck(False)
cdef svm_node **array2d_to_node(np.ndarray[np.float_t, ndim=2] x):
    cdef int i
    cdef svm_node **ret

    ret = <svm_node **> malloc (x.shape[0] * sizeof(svm_node *))
    
    for i in range(x.shape[0]):
        ret[i] = array1d_to_node(x[i], i+1)
            
    return ret

@cython.boundscheck(False)
cdef double *array1d_to_vector(np.ndarray[np.float_t, ndim=1] y):
    cdef int i
    cdef double *ret

    ret = <double *> malloc (y.shape[0] * sizeof(double))
    
    for i in range(y.shape[0]):
        ret[i] = y[i]

    return ret


cdef class LibSvmBase:
    cdef svm_problem problem
    cdef svm_parameter parameter
    cdef svm_model *model
  
    SVM_TYPE = ['c_svc',
                'nu_svc',
                'one_class',
                'epsilon_svr',
                'nu_svr']

    def __cinit__(self, svm_type='c_svc', C=1, nu=0.5, eps=0.001,
                  p=0.1, shrinking=True, probability=False, weight={}):
        """LibSvm.
        
        :Parameters:
            svm_type : string
                SVM type, can be one of: 'c_svc', 'nu_svc', 
                'one_class', 'epsilon_svr', 'nu_svr'.             
            C : float (for 'c_svc', 'epsilon_svr', 'nu_svr')
                cost of constraints violation
            nu : float (for 'nu_svc', 'one_class', 'nu_svr')
                nu parameter
            eps : float
                stopping criterion, usually 0.00001 in nu-SVC,
                0.001 in others
            p : float (for 'epsilon_svr')
                p is the epsilon in epsilon-insensitive loss function
                of epsilon-SVM regression
            shrinking : bool
                use the shrinking heuristics
            probability : bool
                predict probability estimates
            weight : dict 
                changes the penalty for some classes (if the weight for a
                class is not changed, it is set to 1). For example, to
                change penalty for classes 1 and 2 to 0.5 and 0.8
                respectively set weight={1:0.5, 2:0.8}
        """
        
        svm_set_print_string_function(&print_null)

        try:
            self.parameter.svm_type = self.SVM_TYPE.index(svm_type)
        except ValueError:
            raise ValueError("invalid svm_type")
	        
        self.parameter.kernel_type = 4 # precomputed kernel
        self.parameter.degree = 3 # unused
        self.parameter.gamma = 0.001 # unused
        self.parameter.coef0 = 0 # unused
        self.parameter.C = C
        self.parameter.nu = nu
        self.parameter.eps = eps
        self.parameter.p = p
        self.parameter.cache_size = 100 # unused
        self.parameter.shrinking = int(shrinking)
        self.parameter.probability = int(probability)
	
        # weight
        self.parameter.nr_weight = len(weight)
        self.parameter.weight_label = <int *> malloc \
            (len(weight) * sizeof(int))
        self.parameter.weight = <double *> malloc \
            (len(weight) * sizeof(double))
        try:
            for i, key in enumerate(weight):
                self.parameter.weight_label[i] = int(key)
                self.parameter.weight[i] = float(weight[key])
        except ValueError:
            raise ValueError("invalid weight")
        
        self.model = NULL
                    
    def __dealloc__(self):
        self._free_problem()
        self._free_model()
        self._free_param()

    def _load_problem(self, K, y):
        """Convert the data into libsvm svm_problem struct
        """

        Karr = np.ascontiguousarray(K, dtype=np.float)
        yarr = np.ascontiguousarray(y, dtype=np.float)
        
        if Karr.ndim != 2:
            raise ValueError("K must be a 2d array_like object")

        if Karr.shape[0] != Karr.shape[1]:
            raise ValueError("K must be a square matrix")

        if yarr.ndim != 1:
            raise ValueError("y must be an 1d array_like object")

        if Karr.shape[0] != yarr.shape[0]:
            raise ValueError("K, y: shape mismatch")
        
        self.problem.x = array2d_to_node(Karr)
        self.problem.y = array1d_to_vector(yarr)
        self.problem.l = Karr.shape[0]
                
    def learn(self, K, y):
        """Learning method.
        For classification, y is an integer indicating the class label
        (multi-class is supported). For regression, y is the target
        value which can be any real number. For one-class SVM, it's not used
        so can be any number.
        
        :Parameters:
            K : 2d array_like object
                training data in feature space (N, N)
            y : 1d array_like object
                target values (N)
        """
        
        cdef char *ret

        srand(1)
        
        self._free_problem()
        self._load_problem(K, y)
        ret = svm_check_parameter(&self.problem, &self.parameter)
	
        if ret != NULL:
            raise ValueError(ret)

        self._free_model()       
        self.model = svm_train(&self.problem, &self.parameter)
        
    @cython.boundscheck(False)
    def pred(self, Kt):
        """Does classification or regression on test vector(s) Kt.
                
        :Parameters:
            Kt : 1d (one sample) or 2d array_like object ([M], N)
            precomputed test kernel matrix: precomputed inner products 
            (in feature space) between M testing and N training points.
            
        :Returns:
            p : for a classification model, the predicted class(es) for Kt is
                returned. For a regression model, the function value(s) of Kt
                calculated using the model is returned. For an one-class
                model, +1 or -1 is returned.
        """

        cdef int i
        cdef svm_node *test_node

        Ktarr = np.ascontiguousarray(Kt, dtype=np.float)

        if Ktarr.ndim > 2:
            raise ValueError("Kt must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if Ktarr.ndim == 1:
            test_node = array1d_to_node(Ktarr)
            p = svm_predict(self.model, test_node)
            free(test_node)
            if self.SVM_TYPE[self.parameter.svm_type] in \
                    ['c_svc', 'nu_svc', 'one_class']:
                return int(p)
            else:
                return p
        else:
            p = np.empty(Ktarr.shape[0], dtype=np.float)
            for i in range(Ktarr.shape[0]):
                test_node = array1d_to_node(Ktarr[i])
                p[i] = svm_predict(self.model, test_node)
                free(test_node)
        
            if self.SVM_TYPE[self.parameter.svm_type] in \
                    ['c_svc', 'nu_svc', 'one_class']:
                return p.astype(np.int)
            else:       
                return p
    
    @cython.boundscheck(False)
    def pred_values(self, Kt):
        """Returns D decision values for each test sample. 
        For a classification model with C classes, this method
        returns D=C*(C-1)/2 decision values for each test sample. 
        The order is l[0] vs. l[1], ..., l[0] vs. l[C-1], l[1] vs. 
        l[2], ..., l[C-2] vs. l[C-1], where l can be obtained 
        from the labels() method.
        
        For a one-class model, this method returns D=1 decision value 
        for each test sample.
        
        For a regression model, this method returns the predicted
        value as in pred().
                
        :Parameters:
            Kt : 1d (one sample) or 2d array_like object ([M], N)
               precomputed test kernel matrix: precomputed inner products 
               (in feature space) between M testing and N training points.          
        :Returns:
            decision values : 1d (D) or 2d numpy array (M, D)
                decision values for each observation.
        """

        cdef int i, j
        cdef svm_node *test_node
        cdef double *dec_values

        Ktarr = np.ascontiguousarray(Kt, dtype=np.float)

        if Ktarr.ndim > 2:
            raise ValueError("Kt must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")

        if self.SVM_TYPE[self.parameter.svm_type] == 'c_svc' or \
                self.SVM_TYPE[self.parameter.svm_type] == 'nu_svc':
            n = self.model.nr_class*(self.model.nr_class - 1) / 2
        else:
            n = 1

        dec_values = <double *> malloc (n * sizeof(double))
        
        if Ktarr.ndim == 1:
            dec_values_arr = np.empty(n, dtype=np.float)
            test_node = array1d_to_node(Ktarr)
            p = svm_predict_values(self.model, test_node, dec_values)
            free(test_node)
            for j in range(n):
                dec_values_arr[j] = dec_values[j]
        else:
            dec_values_arr = np.empty((Ktarr.shape[0], n), dtype=np.float)
            for i in range(Ktarr.shape[0]):
                test_node = array1d_to_node(Ktarr[i])
                p = svm_predict_values(self.model, test_node, dec_values)
                free(test_node)
                for j in range(n):
                    dec_values_arr[i, j] = dec_values[j]
        
        free(dec_values)
        return dec_values_arr

    @cython.boundscheck(False)
    def pred_probability(self, Kt):
        """Returns C (number of classes) probability estimates.
        For a 'c_svc' and 'nu_svc' classification models with probability 
        information, this method computes 'number of classes' probability 
        estimates.

        :Parameters:
            Kt : 1d (one sample) or 2d array_like object ([M], N)
               precomputed test kernel matrix: precomputed inner products 
               (in feature space) between M testing and N training points.
            
        :Returns:
            probability estimates : 1d (C) or 2d numpy array (M,C)
                probability estimates for each observation.
        """
        
        cdef int i, j
        cdef svm_node *test_node
        cdef double *prob_estimates

        Ktarr = np.ascontiguousarray(Kt, dtype=np.float)

        if Ktarr.ndim > 2:
            raise ValueError("Kt must be an 1d or a 2d array_like object")
        
        if self.model is NULL:
            raise ValueError("no model computed")
        
        if self.SVM_TYPE[self.parameter.svm_type] != 'c_svc' and \
                self.SVM_TYPE[self.parameter.svm_type] != 'nu_svc':
            raise ValueError("probability estimates are available only for"
                             "'c_svc', 'nu_svc' svm types")
        
        ret = svm_check_probability_model(self.model)
        if ret == 0:
            raise ValueError("model does not contain required information"
                             " to do probability estimates. Set probability"
                             "=True")

        prob_estimates = <double*> malloc (self.model.nr_class * 
                                           sizeof(double))
        
        if Ktarr.ndim == 1:
            prob_estimates_arr = np.empty(self.model.nr_class, dtype=np.float)
            test_node = array1d_to_node(Ktarr)
            p = svm_predict_probability(self.model, test_node,
                prob_estimates)
            free(test_node)
            for j in range(self.model.nr_class):
                prob_estimates_arr[j] = prob_estimates[j]
        else:
            prob_estimates_arr = np.empty((Ktarr.shape[0], self.model.nr_class), 
                                          dtype=np.float)
            for i in range(Ktarr.shape[0]):
                test_node = array1d_to_node(Ktarr[i])
                p = svm_predict_probability(self.model, test_node,
                                            prob_estimates)
                free(test_node)
                for j in range(self.model.nr_class):
                    prob_estimates_arr[i, j] = prob_estimates[j]

        free(prob_estimates)
        return prob_estimates_arr
   
    @cython.boundscheck(False)
    def labels(self):
        """For a classification model, this method outputs class
        labels. For regression and one-class models, this method
        returns None.
        """
        
        cdef int i

        if self.model is NULL:
            raise ValueError("no model computed")

        if self.model.label is NULL:
            ret = None
        else:
            ret = np.empty(self.model.nr_class, dtype=np.int)
            for i in range(self.model.nr_class):
                ret[i] = self.model.label[i]
            
        return ret

    @cython.boundscheck(False)
    def sv_idx(self):
        """Returns the support vector indexes.
        """
        
        cdef int i
        
        if self.model is NULL:
            raise ValueError("no model computed")
        
        ret = np.empty(self.model.l, dtype=np.int)
        for i in range(self.model.l):
            ret[i] = self.model.sv_idx[i]
        
        return ret
    
    cdef void _free_problem(self):
        cdef int i
        
        if self.problem.x is not NULL:
            for i in range(self.problem.l):
                free(self.problem.x[i])
            free(self.problem.x)

        if self.problem.y is not NULL:
            free(self.problem.y)
        
    cdef void _free_model(self):
        svm_free_and_destroy_model(&self.model)

    cdef void _free_param(self):
        svm_destroy_param(&self.parameter)
