"""Provides ResNet class for using in CNNs.

ResNet provides interface for building Residual CNNs.
ResNetParams is the parametrization of these CNNs.

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
from net.batchnormlayer import BatchNormLayerParams, BatchNormLayer
from net.convlayer import ConvLayer, ConvLayerParams
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.dropoutlayer import DropoutLayerParams, DropoutLayer
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from net.netbase import NetBase, NetBaseParams
from net.nonlinearitylayer import NonlinearityLayerParams, NonlinearityLayer
from util.theano_helpers import ReLU

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class ResNetParams(NetBaseParams):
    def __init__(self, type=0, nChan=1, wIn=128, hIn=128, batchSize=128, numJoints=16, nDims=3):
        """
        Init the parametrization

        :type type: int
        :param type: type of descr network
        """

        super(ResNetParams, self).__init__()

        self.batch_size = batchSize
        self.numJoints = numJoints
        self.nDims = nDims
        self.numInputs = 1
        self.inputDim = (batchSize, nChan, hIn, wIn)
        self.type = type

        if type == 0:
            # set in net class
            self.numOutputs = 1

            self.outputDim = (batchSize, numJoints * nDims)
        elif type == 1:
            # set in net class
            self.numOutputs = 1

            self.outputDim = (batchSize, numJoints * nDims)
        elif type == 2:
            # set in net class
            self.numOutputs = 1

            self.outputDim = (batchSize, numJoints * nDims)
        elif type == 3:
            # set in net class
            self.numOutputs = 1

            self.outputDim = (batchSize, numJoints * nDims)
        else:
            raise NotImplementedError("not implemented")


class ResNet(NetBase):
    def __init__(self, rng, inputVar=None, cfgParams=None):
        """
        @see https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
        :type cfgParams: DescriptorNetParams
        """
        import theano.tensor as T

        self._params_filter = []
        self._weights_filter = []

        if cfgParams is None:
            raise Exception("Cannot create a Net without config parameters (ie. cfgParams==None)")

        if inputVar is None:
            inputVar = T.tensor4('x')  # input variable
        elif isinstance(inputVar, str):
            raise NotImplementedError()

        self.inputVar = inputVar
        self.cfgParams = cfgParams

        # create network
        self.layers = []

        batchSize = cfgParams.batch_size

        # create structure
        if cfgParams.type == 0:
            # Try ResNet similar configuration
            depth = 47
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001)'
            n = (depth - 2) / 9

            nStages = [32, 64, 128, 256, 256]

            self.layers.append(ConvPoolLayer(rng, self.inputVar,
                                             ConvPoolLayerParams(inputDim=self.cfgParams.inputDim, nFilters=nStages[0],
                                                                 filterDim=(5, 5), stride=(1, 1),
                                                                 poolsize=(2, 2), border_mode='same', activation=None,
                                                                 init_method='He'),
                                             layerNum=len(self.layers)))  # one conv at the beginning
            rout = self.add_res_layers(rng, self.layers[-1].output, self.layers[-1].cfgParams.outputDim, nStages[1], n, 2)  # Stage 1
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[2], n, 2)  # Stage 2
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[3], n, 2)  # Stage 3
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[4], n, 2)  # Stage 4
            self.layers.append(BatchNormLayer(rng, rout, BatchNormLayerParams(inputDim=self.layers[-1].cfgParams.outputDim), layerNum=len(self.layers)))
            self.layers.append(NonlinearityLayer(rng, self.layers[-1].output, NonlinearityLayerParams(inputDim=self.layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(self.layers)))
            # self.layers.append(GlobalAveragePoolLayer(rng, self.layers[-1].output, GlobalAveragePoolLayerParams(inputDim=self.layers[-1].cfgParams.outputDim), layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output.flatten(2),
                                           HiddenLayerParams(# inputDim=self.layers[-1].cfgParams.outputDim,
                                                             inputDim=(self.layers[-1].cfgParams.outputDim[0], numpy.prod(self.layers[-1].cfgParams.outputDim[1:])),
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, self.cfgParams.numJoints*self.cfgParams.nDims),
                                                             activation=None),
                                           layerNum=len(self.layers)))

            self.output = self.layers[-1].output
        elif cfgParams.type == 1:
            # Try ResNet similar configuration
            depth = 47
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001)'
            n = (depth - 2) / 9

            nStages = [32, 64, 128, 256, 256]

            self.layers.append(ConvPoolLayer(rng, self.inputVar,
                                             ConvPoolLayerParams(inputDim=self.cfgParams.inputDim, nFilters=nStages[0],
                                                                 filterDim=(5, 5), stride=(1, 1),
                                                                 poolsize=(2, 2), border_mode='same', activation=None,
                                                                 init_method='He'),
                                             layerNum=len(self.layers)))  # one conv at the beginning
            rout = self.add_res_layers(rng, self.layers[-1].output, self.layers[-1].cfgParams.outputDim, nStages[1], n, 2)  # Stage 1
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[2], n, 2)  # Stage 2
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[3], n, 2)  # Stage 3
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[4], n, 2)  # Stage 4
            self.layers.append(BatchNormLayer(rng, rout, BatchNormLayerParams(inputDim=self.layers[-1].cfgParams.outputDim), layerNum=len(self.layers)))
            self.layers.append(NonlinearityLayer(rng, self.layers[-1].output, NonlinearityLayerParams(inputDim=self.layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output.flatten(2),
                                           HiddenLayerParams(inputDim=(self.layers[-1].cfgParams.outputDim[0], numpy.prod(self.layers[-1].cfgParams.outputDim[1:])),
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, 30), activation=None),
                                           layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, self.cfgParams.numJoints*self.cfgParams.nDims),
                                                             activation=None),
                                           layerNum=len(self.layers)))

            self.output = self.layers[-1].output
        elif cfgParams.type == 2:
            # Try ResNet similar configuration
            depth = 47
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001)'
            n = (depth - 2) / 9

            nStages = [32, 64, 128, 256, 256]

            self.layers.append(ConvPoolLayer(rng, self.inputVar,
                                             ConvPoolLayerParams(inputDim=self.cfgParams.inputDim, nFilters=nStages[0],
                                                                 filterDim=(5, 5), stride=(1, 1),
                                                                 poolsize=(2, 2), border_mode='same', activation=None,
                                                                 init_method='He'),
                                             layerNum=len(self.layers)))  # one conv at the beginning
            rout = self.add_res_layers(rng, self.layers[-1].output, self.layers[-1].cfgParams.outputDim, nStages[1], n, 2)  # Stage 1
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[2], n, 2)  # Stage 2
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[3], n, 2)  # Stage 3
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[4], n, 2)  # Stage 4
            self.layers.append(BatchNormLayer(rng, rout, BatchNormLayerParams(inputDim=self.layers[-1].cfgParams.outputDim), layerNum=len(self.layers)))
            self.layers.append(NonlinearityLayer(rng, self.layers[-1].output, NonlinearityLayerParams(inputDim=self.layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output.flatten(2),
                                           HiddenLayerParams(inputDim=(self.layers[-1].cfgParams.outputDim[0], numpy.prod(self.layers[-1].cfgParams.outputDim[1:])),
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))

            self.layers.append(DropoutLayer(rng, self.layers[-1].output,
                                            DropoutLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                               outputDim=self.layers[-1].cfgParams.outputDim),
                                            layerNum=len(self.layers)))

            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))

            self.layers.append(DropoutLayer(rng, self.layers[-1].output,
                                            DropoutLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                               outputDim=self.layers[-1].cfgParams.outputDim),
                                            layerNum=len(self.layers)))

            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, self.cfgParams.numJoints*self.cfgParams.nDims),
                                                             activation=None),
                                           layerNum=len(self.layers)))

            self.output = self.layers[-1].output
        elif cfgParams.type == 3:
            # Try ResNet similar configuration
            depth = 47
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001)'
            n = (depth - 2) / 9

            nStages = [32, 64, 128, 128, 128]

            self.layers.append(ConvPoolLayer(rng, self.inputVar,
                                             ConvPoolLayerParams(inputDim=self.cfgParams.inputDim, nFilters=nStages[0],
                                                                 filterDim=(5, 5), stride=(1, 1),
                                                                 poolsize=(2, 2), border_mode='same', activation=None,
                                                                 init_method='He'),
                                             layerNum=len(self.layers)))  # one conv at the beginning
            rout = self.add_res_layers(rng, self.layers[-1].output, self.layers[-1].cfgParams.outputDim, nStages[1], n, 2)  # Stage 1
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[2], n, 2)  # Stage 2
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[3], n, 2)  # Stage 3
            rout = self.add_res_layers(rng, rout, self.layers[-1].cfgParams.outputDim, nStages[4], n, 2)  # Stage 4
            self.layers.append(BatchNormLayer(rng, rout, BatchNormLayerParams(inputDim=self.layers[-1].cfgParams.outputDim), layerNum=len(self.layers)))
            self.layers.append(NonlinearityLayer(rng, self.layers[-1].output, NonlinearityLayerParams(inputDim=self.layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(self.layers)))
            self.layers.append(HiddenLayer(rng, self.layers[-1].output.flatten(2),
                                           HiddenLayerParams(inputDim=(self.layers[-1].cfgParams.outputDim[0], numpy.prod(self.layers[-1].cfgParams.outputDim[1:])),
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))

            self.layers.append(DropoutLayer(rng, self.layers[-1].output,
                                            DropoutLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                               outputDim=self.layers[-1].cfgParams.outputDim),
                                            layerNum=len(self.layers)))

            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, 1024), activation=ReLU),
                                           layerNum=len(self.layers)))

            self.layers.append(DropoutLayer(rng, self.layers[-1].output,
                                            DropoutLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                               outputDim=self.layers[-1].cfgParams.outputDim),
                                            layerNum=len(self.layers)))

            self.layers.append(HiddenLayer(rng, self.layers[-1].output,
                                           HiddenLayerParams(inputDim=self.layers[-1].cfgParams.outputDim,
                                                             outputDim=(batchSize, self.cfgParams.numJoints*self.cfgParams.nDims),
                                                             activation=None),
                                           layerNum=len(self.layers)))

            self.output = self.layers[-1].output
        else:
            raise NotImplementedError()

        self.load(self.cfgParams.loadFile)

    def add_res_layers(self, rng, inputVar, inputDim, outputFilters, count, stride):
        rout = res_block(self.layers, rng, inputVar, inputDim, outputFilters, stride)
        for i in xrange(1, count):
            rout = res_block(self.layers, rng, rout, self.layers[-1].cfgParams.outputDim, outputFilters, 1)
        return rout


def res_block(layers, rng, inputVar, inputDim, outputFilters, stride, nBottleneckFilters=None):
    if nBottleneckFilters is None:
        nBottleneckFilters = outputFilters // 4

    if inputDim[1] == outputFilters:
        # conv1x1
        layers.append(BatchNormLayer(rng, inputVar, BatchNormLayerParams(inputDim=inputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=nBottleneckFilters, filterDim=(1, 1),
                                                                        border_mode='same', activation=None,
                                                                        init_method='He'), layerNum=len(layers)))

        # conv3x3
        layers.append(BatchNormLayer(rng, layers[-1].output, BatchNormLayerParams(inputDim=layers[-1].cfgParams.outputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=nBottleneckFilters, filterDim=(3, 3),
                                                                        border_mode='same', activation=None,
                                                                        init_method='He'), layerNum=len(layers)))

        # conv1x1
        layers.append(BatchNormLayer(rng, layers[-1].output, BatchNormLayerParams(inputDim=layers[-1].cfgParams.outputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=outputFilters, filterDim=(1, 1),
                                                                        border_mode='same', activation=None,
                                                                        init_method='He'), layerNum=len(layers)))

        # add identity connection
        return inputVar + layers[-1].output
    else:
        # common BN, ReLU
        layers.append(BatchNormLayer(rng, inputVar, BatchNormLayerParams(inputDim=inputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))

        # conv1x1
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=nBottleneckFilters, filterDim=(1, 1),
                                                                        stride=(stride, stride), border_mode='same',
                                                                        activation=None, init_method='He'), layerNum=len(layers)))

        # conv3x3
        layers.append(BatchNormLayer(rng, layers[-1].output, BatchNormLayerParams(inputDim=layers[-1].cfgParams.outputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=nBottleneckFilters, filterDim=(3, 3),
                                                                        border_mode='same', activation=None,
                                                                        init_method='He'), layerNum=len(layers)))

        # conv1x1
        layers.append(BatchNormLayer(rng, layers[-1].output, BatchNormLayerParams(inputDim=layers[-1].cfgParams.outputDim), layerNum=len(layers)))
        layers.append(NonlinearityLayer(rng, layers[-1].output, NonlinearityLayerParams(inputDim=layers[-1].cfgParams.outputDim, activation=ReLU), layerNum=len(layers)))
        layers.append(ConvLayer(rng, layers[-1].output, ConvLayerParams(inputDim=layers[-1].cfgParams.outputDim,
                                                                        nFilters=outputFilters, filterDim=(1, 1),
                                                                        border_mode='same', activation=None,
                                                                        init_method='He'), layerNum=len(layers)))

        # shortcut
        layers.append(ConvLayer(rng, layers[-8].output, ConvLayerParams(inputDim=layers[-8].cfgParams.outputDim,
                                                                        nFilters=outputFilters, filterDim=(1, 1),
                                                                        stride=(stride, stride), border_mode='same',
                                                                        activation=None, init_method='He'), layerNum=len(layers)))

        # add identity connection
        return layers[-2].output + layers[-1].output
