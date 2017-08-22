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
        self.updates = []
        self.shared = []

        if len(grads) != len(params):
            print "Warning: Size of gradients ({}) does not fit size of parameters ({})!".format(len(grads), len(params))

    def ADAM(self, learning_rate=0.0002, beta1=0.9, beta2=0.999, epsilon=1e-8, gamma=1-1e-8):
        """
        Adam update rule by Kingma and Ba, ICLR 2015, version 2 (with momentum decay).
        :param learning_rate: alpha in the paper, the step size
        :param beta1: exponential decay rate of the 1st moment estimate
        :param beta2: exponential decay rate of the 2nd moment estimate
        :param epsilon: small epsilon to prevent divide-by-0 errors
        :param gamma: exponential increase rate of beta1
        :return: updates
        """

        t = theano.shared(numpy.cast[theano.config.floatX](1.0))  # timestep, for bias correction
        beta1_t = beta1*gamma**(t-1.)  # decay the first moment running average coefficient

        for param_i, grad_i in zip(self.params, self.grads):
            mparam_i = theano.shared(numpy.zeros(param_i.get_value().shape, dtype=theano.config.floatX))  # 1st moment
            self.shared.append(mparam_i)
            vparam_i = theano.shared(numpy.zeros(param_i.get_value().shape, dtype=theano.config.floatX))  # 2nd moment
            self.shared.append(vparam_i)

            m = beta1_t * mparam_i + (1. - beta1_t) * grad_i  # new value for 1st moment estimate
            v = beta2 * vparam_i + (1. - beta2) * T.sqr(grad_i)  # new value for 2nd moment estimate

            m_unbiased = m / (1. - beta1**t)  # bias corrected 1st moment estimate
            v_unbiased = v / (1. - beta2**t)  # bias corrected 2nd moment estimate
            w = param_i - (learning_rate * m_unbiased) / (T.sqrt(v_unbiased) + epsilon)  # new parameter values

            self.updates.append((mparam_i, m))
            self.updates.append((vparam_i, v))
            self.updates.append((param_i, w))
        self.updates.append((t, t + 1.))

        return self.updates

    def RMSProp(self, learning_rate=0.01, decay=0.9, epsilon=1.0 / 100.):
        """
        RMSProp of Tieleman et al.
        :param learning_rate: learning rate
        :param decay: decay rate of gradient history
        :param epsilon: gradient clip
        :return: update
        """

        for param_i, grad_i in zip(self.params, self.grads):
            # Accumulate gradient
            msg = theano.shared(numpy.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
            self.shared.append(msg)
            new_mean_squared_grad = (decay * msg + (1 - decay) * T.sqr(grad_i))

            # Compute update
            rms_grad_t = T.sqrt(new_mean_squared_grad)
            rms_grad_t = T.maximum(rms_grad_t, epsilon)
            delta_x_t = -learning_rate * grad_i / rms_grad_t

            # Apply update
            self.updates.append((param_i, param_i + delta_x_t))
            self.updates.append((msg, new_mean_squared_grad))

        return self.updates

