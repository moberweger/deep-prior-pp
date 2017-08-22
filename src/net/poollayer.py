"""Provides PoolLayer class for using in CNNs.

PoolLayer provides interface for building pooling layers in CNNs.
PoolLayerParams is the parametrization of these PoolLayer layers.

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
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class PoolLayerParams(LayerParams):
    
    def __init__(self, inputDim=None, poolsize=None, redDim=None, outputDim=None, activation=None, poolType=0):
        """

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        super(PoolLayerParams, self).__init__(inputDim, outputDim)

        self._poolsize = poolsize
        self._redDim = redDim
        self._activation = activation
        self._poolType = poolType
        self.update()

    @property 
    def poolsize(self):
        return self._poolsize
    
    @poolsize.setter 
    def poolsize(self, value):
        self._poolsize = value
        self.update()

    @property
    def activation(self):
        return self._activation

    @property
    def poolType(self):
        return self._poolType
                     
    def update(self):
        """
        calc image_shape, 
        """
        self._outputDim = (self._inputDim[0],   # batch_size
                           self._redDim*self._inputDim[1] if self._redDim is not None else self._inputDim[1],      # number of kernels
                           self._inputDim[2]//self._poolsize[0],   #  output H
                           self._inputDim[3]//self._poolsize[1])   #  output W

        if(self._poolsize[0] == 1) and (self._poolsize[1] == 1):
            self._poolType = -1


class PoolLayer(Layer):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, rng, inputVar, cfgParams, copyLayer=None, layerNum=None):
        """
        Allocate a PoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputVar: theano.tensor.dtensor4
        :param inputVar: symbolic image tensor, of shape image_shape

        :type cfgParams: PoolLayerParams
        """
        import theano
        import theano.sandbox.neighbours
        import theano.tensor as T
        from theano.tensor.signal.pool import pool_2d

        super(PoolLayer, self).__init__(rng)

        floatX = theano.config.floatX  # @UndefinedVariable

        outputDim = cfgParams.outputDim
        poolsize = cfgParams.poolsize
        inputDim = cfgParams.inputDim
        activation = cfgParams.activation
        poolType = cfgParams.poolType

        self.cfgParams = cfgParams
        self.layerNum = layerNum

        self.inputVar = inputVar

        if inputVar.type.ndim != 4:
            raise TypeError()

        self.params = []
        self.weights = []

        # downsample each feature map individually, using maxpooling
        if poolType == 0:
            # use maxpooling
            pooled_out = pool_2d(input=self.inputVar, ds=poolsize, ignore_border=True, mode='max')
        elif poolType == 1:
            # use average pooling
            pooled_out = pool_2d(input=self.inputVar, ds=poolsize, ignore_border=True, mode='average_inc_pad')
        elif poolType == 3:
            # use subsampling and ignore border
            pooled_out = self.inputVar[:, :, :(inputDim[2]//poolsize[0])*poolsize[0], :(inputDim[3]//poolsize[1])*poolsize[1]][:, :, ::poolsize[0], ::poolsize[1]]
        elif poolType == -1:
            # no pooling at all
            pooled_out = self.inputVar
        else:
            raise NotImplementedError()
        self.output_pre_act = pooled_out

        self.output = (pooled_out if activation is None
                       else activation(pooled_out))

        self.output.name = 'output_layer_{}'.format(self.layerNum)

    def __str__(self):
        """
        Print configuration of layer
        :return: configuration string
        """
        return "poolsize {}, pooltype {}, activation {}".format(self.cfgParams.poolsize, self.cfgParams.poolType,
                                                                self.cfgParams.activation_str)
