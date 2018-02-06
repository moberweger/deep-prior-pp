"""Provides Layer class used as base for other layers.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy
from util.theano_helpers import ReLU, sigmoid, tanh

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class Layer(object):
    """
    Base for different layers for CNNs
    """

    def __init__(self, rng):
        """
        Constructor
        """        
        self.weights = []
        self.params = []
        self.params_nontrained = []
        self.rng = rng

    def orthogonalize(self, init_vals):
        # try pca to create an orthogonal set of filters to start with
        wInitVals = numpy.reshape(init_vals, (init_vals.shape[0], -1))
        svd = numpy.linalg.svd(wInitVals.T)
        U = svd[0]
        wInitVals = U.T[0:init_vals.shape[0]].T
        return numpy.reshape(wInitVals.swapaxes(0, 1), init_vals.shape)

    def getOptimalInitMethod(self, act_str):

        if act_str == ReLU.__name__:
            return 'He'
        elif act_str == sigmoid.__name__:
            return 'sigmoid'
        elif act_str in tanh.__name__:
            return 'tanh'
        elif act_str is None or str(act_str) == 'None':
            return None
        else:
            raise NotImplementedError("Unknown activation function: {}".format(act_str))

    def getInitVals(self, shape, mode, act_fn=None, method=None, orthogonal=False):
        import theano

        floatX = theano.config.floatX  # @UndefinedVariable

        if act_fn is None and method is None:
            raise UserWarning("act_fn and method not defined! At least one must be specified.")

        if act_fn is not None and method is None:
            method = self.getOptimalInitMethod(act_fn)

        # initialize weights with random weights
        if method == 'He':
            # Initialization of He, Zhang, Ren and Sun, Delving Deep into Rectifiers, 2015
            if mode == 'conv':
                W_bound = numpy.sqrt(2. / numpy.prod(shape[1:]))
                init_vals = numpy.asarray(self.rng.normal(loc=0.0, scale=W_bound, size=shape), dtype=floatX)
            elif mode == 'fc':
                init_vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.01, size=shape), dtype=floatX)
            else:
                raise NotImplementedError()
        elif method == 'Xavier':
            # Initialization "Xavier" of Glorot and Bengio, Understanding the difficulty of training deep feedforward neural networks, 2010
            if mode == 'conv':
                W_bound = numpy.sqrt(3. / numpy.prod(shape[1:]))
                init_vals = numpy.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=shape), dtype=floatX)
            elif mode == 'fc':
                W_bound = numpy.sqrt(1. / shape[0])
                init_vals = numpy.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=shape), dtype=floatX)
            else:
                raise NotImplementedError()
        elif method == 'sigmoid':
            if mode == 'conv':
                W_bound = 4. * numpy.sqrt(6. / (numpy.prod(shape[1:]) + (shape[0] * numpy.prod(shape[2:]))))
                init_vals = numpy.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=shape), dtype=floatX)
            elif mode == 'fc':
                init_vals = 4. * numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / numpy.sum(shape)),
                                                                high=numpy.sqrt(6. / numpy.sum(shape)),
                                                                size=shape), dtype=floatX)
            else:
                raise NotImplementedError()
        elif method == 'tanh' or method is None:
            if mode == 'conv':
                W_bound = 1. / (numpy.prod(shape[1:]) + (shape[0] * numpy.prod(shape[2:])))
                init_vals = numpy.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=shape), dtype=floatX)
            elif mode == 'fc':
                init_vals = numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / numpy.sum(shape)),
                                                           high=numpy.sqrt(6. / numpy.sum(shape)),
                                                           size=shape), dtype=floatX)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError("Unknown method!")

        return init_vals if orthogonal is False else self.orthogonalize(init_vals)
