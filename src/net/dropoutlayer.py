"""Provides DropoutLayer class for using in CNNs.

DropoutLayer provides interface for building dropout layers in CNNs.
DropoutLayerParams is the parametrization of these DropoutLayer layers.

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


class DropoutLayerParams(LayerParams):
    def __init__(self, inputDim=None, outputDim=None, p=0.3):
        """
        :type inputDim: tuple of [int]
        :param inputDim: dimensionality of input

        :type outputDim: tuple of [int]
        :param outputDim: number of hidden units

        :type p: float
        :param p: Probability for dropping a unit of the layer
        """

        super(DropoutLayerParams, self).__init__(inputDim=inputDim, outputDim=outputDim)

        self._p = p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value


class DropoutLayer(Layer):
    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Dropout layer of a MLP: units are fully-connected and connections are
        dropped randomly during training.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize mask

        :type inputVar: theano.tensor.matrix
        :param inputVar: a symbolic tensor of shape (n_examples, n_in)

        :type cfgParams: DropoutLayerParams
        """
        import theano
        import theano.tensor as T
        from theano.ifelse import ifelse

        super(DropoutLayer, self).__init__(rng)

        self.inputVar = inputVar
        self.cfgParams = cfgParams
        self.layerNum = layerNum

        assert 0. < cfgParams.p < 1.

        # see https://github.com/uoguelph-mlrg/theano_alexnet/blob/master/alex_net.py
        self.prob_drop = cfgParams.p
        self.prob_keep = 1.0 - cfgParams.p
        self.flag_on = theano.shared(numpy.cast[theano.config.floatX](1.0), name='flag_on')

        # mask_rng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        # faster rng on GPU
        from theano.sandbox.rng_mrg import MRG_RandomStreams
        mask_rng = MRG_RandomStreams(rng.randint(999999))
        if not (copyLayer is None):
            self.mask = copyLayer.mask
        else:
            self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=self.cfgParams.inputDim, dtype=theano.config.floatX)
        self.output = ifelse(T.gt(self.flag_on, 0), self.mask * self.inputVar, self.prob_keep * self.inputVar)
        self.output.name = 'output_layer_{}'.format(self.layerNum)
        self.output_pre_act = self.output  # for compatibility

        # no params and weights
        self.params = []
        self.weights = []

    def unsetDeterministic(self):
        """
        Enable dropout
        :return: None
        """
        self.flag_on.set_value(1.0)

    def setDeterministic(self):
        """
        Disable dropout
        :return: None
        """
        self.flag_on.set_value(0.0)

    def isDeterministic(self):
        """
        Check if dropout is enabled
        :return: True if enabled
        """
        return bool(numpy.allclose(self.flag_on.get_value(), 0.0))

    def __str__(self):
        """
        Print configuration of layer
        :return: configuration string
        """
        return "inputDim {}, outputDim {}, p {}".format(self.cfgParams.inputDim, self.cfgParams.outputDim, self.cfgParams.p)
