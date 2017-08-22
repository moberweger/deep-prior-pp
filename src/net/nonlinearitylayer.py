"""Provides NonlinearityLayer class for using in CNNs.

NonlinearityLayer provides interface for applying a activation
function to the output of a layer for CNNs.
NonlinearityLayerParams is the parametrization of these
NonlinearityLayer layers.

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

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class NonlinearityLayerParams(LayerParams):
    def __init__(self, inputDim=None, outputDim=None, activation=None):
        """
        :type inputDim: tuple of [int]
        :param inputDim: dimensionality of input

        :type outputDim: tuple of [int]
        :param outputDim: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """

        super(NonlinearityLayerParams, self).__init__(inputDim, outputDim)

        self._outputDim = self._inputDim
        self._activation = activation

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        self._activation = value

    def getMemoryRequirement(self):
        """
        Get memory requirements of weights
        :return: memory requirement
        """
        return 0


class NonlinearityLayer(Layer):
    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Activation layer that applies non-linearity to input

        Hidden unit activation is given by: activation(inputVar)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dmatrix
        :param inputVar: a symbolic tensor of shape (n_examples, n_in)

        :type cfgParams: HiddenLayerParams
        """
        import theano

        super(NonlinearityLayer, self).__init__(rng)

        assert isinstance(cfgParams, NonlinearityLayerParams)

        self.inputVar = inputVar
        self.cfgParams = cfgParams
        self.layerNum = layerNum

        n_out = cfgParams.outputDim[1]
        activation = cfgParams.activation

        floatX = theano.config.floatX  # @UndefinedVariable
        self.output_pre_act = inputVar

        if activation is None:
            self.output = inputVar
            self.output.name = 'output_layer_{}'.format(self.layerNum)
            self.params = []
        else:
            if inspect.isfunction(activation) and len(inspect.getargspec(activation).args) == 2:
                c_values = numpy.ones((n_out,), dtype=floatX)*0.5
                self.c = theano.shared(value=c_values, name='c{}'.format(layerNum), borrow=True)
                self.output = activation(inputVar, self.c)
                self.output.name = 'output_layer_{}'.format(self.layerNum)
                self.params = [self.c]
            else:
                self.output = activation(inputVar)
                self.output.name = 'output_layer_{}'.format(self.layerNum)
                self.params = []

        # parameters of the model
        self.weights = []

    def __str__(self):
        """
        Print configuration of layer
        :return: configuration string
        """
        return "inputDim {}, outputDim {}, activation {}".format(self.cfgParams.inputDim,
                                                                 self.cfgParams.outputDim,
                                                                 self.cfgParams.activation_str)
