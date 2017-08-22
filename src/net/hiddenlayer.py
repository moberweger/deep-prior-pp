"""Provides HiddenLayer class for using in CNNs.

HiddenLayer provides interface for building hidden (fully connected) layers in CNNs.
HiddenLayerParams is the parametrization of these HiddenLayer layers.

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

import inspect
import numpy
from net.layer import Layer
from net.layerparams import LayerParams

__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>, Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HiddenLayerParams(LayerParams):
    def __init__(self, inputDim=None, outputDim=None, activation=None, hasBias=True, init_method=None):
        """
        :type inputDim: tuple of [int]
        :param inputDim: dimensionality of input

        :type outputDim: tuple of [int]
        :param outputDim: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """

        super(HiddenLayerParams, self).__init__(inputDim, outputDim)

        self._activation = activation
        self._hasbias = hasBias
        self._init_method = init_method

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    @property
    def hasBias(self):
        return self._hasbias

    @hasBias.setter
    def hasBias(self, value):
        self._hasbias = value

    def getMemoryRequirement(self):
        """
        Get memory requirements of weights
        :return: memory requirement
        """
        return ((self.inputDim[1] * self.outputDim[1]) + self.outputDim[1]) * 4  # sizeof(theano.config.floatX)


class HiddenLayer(Layer):
    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Typical hidden layer of a MLP: units are fully-connected.
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: activation(dot(inputVar,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dmatrix
        :param inputVar: a symbolic tensor of shape (n_examples, n_in)

        :type cfgParams: HiddenLayerParams
        """
        import theano
        import theano.tensor as T

        super(HiddenLayer, self).__init__(rng)

        assert isinstance(cfgParams, HiddenLayerParams)

        self.inputVar = inputVar
        self.cfgParams = cfgParams
        self.layerNum = layerNum

        n_in = cfgParams.inputDim[1]
        n_out = cfgParams.outputDim[1]
        activation = cfgParams.activation

        # `W` is initialized with `W_values` which is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and
        # sqrt(6./(n_in+n_hidden)) for tanh activation function the output of uniform if converted using asarray
        # to dtype theano.config.floatX so that the code is runable on GPU.
        # Note : optimal initialization of weights is dependent on the activation function used (among other things).
        # For example, results presented in [Xavier10] suggest that you should use 4 times larger initial weights for
        # sigmoid compared to tanh. We have no info for other function, so we use the same as tanh.
        floatX = theano.config.floatX  # @UndefinedVariable

        if copyLayer is None:
            wInitVals = self.getInitVals((n_in, n_out), 'fc', act_fn=cfgParams.activation_str, method=cfgParams._init_method)
            self.W = theano.shared(value=wInitVals, name='W{}'.format(layerNum), borrow=True)

            if self.cfgParams.hasBias is True:
                b_values = numpy.zeros((n_out,), dtype=floatX)
                self.b = theano.shared(value=b_values, name='b{}'.format(layerNum), borrow=True)

        else:
            self.W = copyLayer.W
            if self.cfgParams.hasBias is True:
                self.b = copyLayer.b

        lin_output = T.dot(inputVar, self.W)

        if self.cfgParams.hasBias is True:
            lin_output = lin_output + self.b
        self.output_pre_act = lin_output

        if activation is None:
            self.output = lin_output
            self.output.name = 'output_layer_{}'.format(self.layerNum)
            self.params = [self.W, self.b] if self.cfgParams.hasBias else [self.W]
        else:
            if inspect.isfunction(activation) and len(inspect.getargspec(activation).args) == 2:
                c_values = numpy.ones((n_out,), dtype=floatX)*0.5
                self.c = theano.shared(value=c_values, name='c{}'.format(layerNum), borrow=True)
                self.output = activation(lin_output, self.c)
                self.output.name = 'output_layer_{}'.format(self.layerNum)
                self.params = [self.W, self.b, self.c] if self.cfgParams.hasBias else [self.W, self.c]
            else:
                self.output = activation(lin_output)
                self.output.name = 'output_layer_{}'.format(self.layerNum)
                self.params = [self.W, self.b] if self.cfgParams.hasBias else [self.W]

        # parameters of the model
        self.weights = [self.W]

    def __str__(self):
        """
        Print configuration of layer
        :return: configuration string
        """
        return "inputDim {}, outputDim {}, activiation {}, hasBias {}".format(self.cfgParams.inputDim,
                                                                              self.cfgParams.outputDim,
                                                                              self.cfgParams.activation_str,
                                                                              self.cfgParams.hasBias)
