"""Provides BatchNormLayer class for using in CNNs.

BatchNormLayer provides interface for building batch normalization layers in CNNs.
These are used to normalize the activations for each batch.
BatchNormLayerParams is the parametrization of these BatchNormLayer layers.

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
from net.layer import Layer
from net.layerparams import LayerParams

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class BatchNormLayerParams(LayerParams):

    def __init__(self, inputDim=None, outputDim=None, epsilon=1e-4, alpha=0.1, mode='low_mem',
                 learn_beta=True, learn_gamma=True):
        """

        :type epsilon: float
        :param epsilon: used for numerical stability
        """

        super(BatchNormLayerParams, self).__init__(inputDim, outputDim)

        self._learn_beta = learn_beta
        self._learn_gamma = learn_gamma
        self._epsilon = epsilon
        self._alpha = alpha
        self._mode = mode
        self._outputDim = self._inputDim

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value


class BatchNormLayer(Layer):
    """
    BatchNormLayer of a convolutional network.
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Ioffe and Szegedy
    @see https://github.com/f0k/Lasagne/blob/batchnorm/lasagne/layers/normalization.py
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Allocate a BatchNormLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: BatchNormLayerParams
        """
        import theano
        import theano.tensor as T

        super(BatchNormLayer, self).__init__(rng)

        floatX = theano.config.floatX  # @UndefinedVariable

        self.cfgParams = cfgParams
        self.layerNum = layerNum
        self.inputVar = inputVar

        inputDim = cfgParams.inputDim
        epsilon = cfgParams.epsilon
        alpha = cfgParams.alpha
        mode = cfgParams.mode

        self.flag_on = theano.shared(numpy.cast[theano.config.floatX](1.0), name='flag_on')

        # normalize over all but the second axis
        axes = (0,) + tuple(range(2, len(inputDim)))

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(inputDim) if axis not in axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for all axes not normalized over.")

        self.beta = theano.shared(numpy.zeros(shape, dtype=floatX), name='beta{}'.format(layerNum), borrow=True)
        self.gamma = theano.shared(numpy.ones(shape, dtype=floatX), name='gamma{}'.format(layerNum), borrow=True)
        self.mean = theano.shared(numpy.zeros(shape, dtype=floatX), name='mean{}'.format(layerNum), borrow=True)
        self.inv_std = theano.shared(numpy.ones(shape, dtype=floatX), name='inv_std{}'.format(layerNum), borrow=True)
        self.weights = []
        self.params = []
        if self.cfgParams._learn_beta is True:
            self.params.append(self.beta)
        if self.cfgParams._learn_gamma is True:
            self.params.append(self.gamma)
        self.params_nontrained = [self.mean, self.inv_std]

        input_mean = T.mean(inputVar, axis=axes)
        input_inv_std = T.inv(T.sqrt(T.var(inputVar, axis=axes) + epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        mean = T.switch(self.flag_on, input_mean, self.mean)
        inv_std = T.switch(self.flag_on, input_inv_std, self.inv_std)

        # Decide whether to update the stored averages
        # Trick: To update the stored statistics, we create memory-aliased
        # clones of the stored statistics:
        running_mean = theano.clone(self.mean, share_inputs=False)
        running_inv_std = theano.clone(self.inv_std, share_inputs=False)
        # set a default update for them:
        running_mean.default_update = T.switch(self.flag_on,
                                               ((1. - alpha) * running_mean + alpha * input_mean),
                                               self.mean)
        running_inv_std.default_update = T.switch(self.flag_on,
                                                  ((1. - alpha) * running_inv_std + alpha * input_inv_std),
                                                  self.inv_std)
        # and make sure they end up in the graph without participating in
        # the computation (this way their default_update will be collected
        # and applied, but the computation will be optimized away):
        mean += 0 * running_mean
        inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(inputVar.ndim - len(axes)))
        pattern = ['x' if input_axis in axes else next(param_axes) for input_axis in range(inputVar.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = self.beta.dimshuffle(pattern)
        gamma = self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        # self.output = (inputVar - mean) * (gamma * inv_std) + beta
        # where either inv_std == 1 or std == 1, depending on which one is used
        self.output = T.nnet.batch_normalization(inputVar, gamma=gamma*inv_std, beta=beta, mean=mean, std=1, mode=mode)
        self.output_pre_act = self.output

    def unsetDeterministic(self):
        """
        Enable batch normalization, learn mean and variance
        :return: None
        """
        self.flag_on.set_value(1.0)

    def setDeterministic(self):
        """
        Disable batch normalization, use learned mean and variance
        :return: None
        """
        self.flag_on.set_value(0.0)

    def isDeterministic(self):
        """
        Check if batch normalization is enabled
        :return: True if enabled
        """
        return bool(numpy.allclose(self.flag_on.get_value(), 0.0))

    def __str__(self):
        """
        Print configuration of layer
        :return: configuration string
        """
        return "epsilon {}, alpha {}".format(self.cfgParams.epsilon, self.cfgParams.alpha)
