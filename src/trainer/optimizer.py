"""Basis for different optimization algorithms.

Optimizer provides interface for creating the update rules for gradient based optimization.
It includes SGD, NAG, RMSProp, etc.

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

import theano
import theano.tensor as T
import numpy

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class Optimizer(object):
    """
    Class with different optimizers of the loss function
    """

    def __init__(self, grads, params):
        """
        Initialize object
        :param grads: gradients of the loss function
        :param params: model parameters that should be updated
        """
        self.grads = grads
        self.params = params

    def RMSProp(self, learning_rate=0.01, decay=0.9, epsilon=1.0 / 100.):
        """
        RMSProp of Tieleman et al.
        :param learning_rate: learning rate
        :param decay: decay rate of gradient history
        :param epsilon: gradient clip
        :return: update
        """

        updates = []

        for param_i, grad_i in zip(self.params, self.grads):
            # Accumulate gradient
            msg = theano.shared(numpy.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
            new_mean_squared_grad = (decay * msg + (1 - decay) * T.sqr(grad_i))

            # Compute update
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, epsilon)
            delta_x_t = -learning_rate * grad_i / rms_grad_t

            # Apply update
            updates.append((param_i, param_i + delta_x_t))
            updates.append((msg, new_mean_squared_grad))

        return updates

