"""Provides PoseRegNet class that implements deep CNNs.

PoseRegNet provides interface for building the CNN.
PoseRegNetParams is the parametrization of these CNNs.

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

import theano.tensor as T
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from net.dropoutlayer import DropoutLayer, DropoutLayerParams
from net.netbase import NetBase, NetBaseParams
from net.poollayer import PoolLayerParams
from util.helpers import ReLU

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class PoseRegNetParams(NetBaseParams):
    def __init__(self, type=0, nChan=1, wIn=128, hIn=128, batchSize=128, numJoints=16, nDims=3):
        """
        Init the parametrization

        :type type: int
        :param type: type of descr network
        """

        super(PoseRegNetParams, self).__init__()

        self.batch_size = batchSize
        self.numJoints = numJoints
        self.nDims = nDims
        self.inputDim = (batchSize, nChan, hIn, wIn)

        if type == 0:
            # Try DeepPose CNN similar configuration
            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, nChan, hIn, wIn),  # w,h,nChannel
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(4, 4),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(2, 2),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(3, 3),
                                               poolsize=(1, 1),
                                               activation=ReLU))

            l3out = self.layers[-1].outputDim
            self.layers.append(HiddenLayerParams(inputDim=(l3out[0], l3out[1] * l3out[2] * l3out[3]),
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(DropoutLayerParams(inputDim=self.layers[-1].outputDim,
                                                  outputDim=self.layers[-1].outputDim))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(DropoutLayerParams(inputDim=self.layers[-1].outputDim,
                                                  outputDim=self.layers[-1].outputDim))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, numJoints * nDims),
                                                 activation=None))

            self.outputDim = self.layers[-1].outputDim
        elif type == 11:
            # Try DeepPose CNN similar configuration
            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, nChan, hIn, wIn),  # w,h,nChannel
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(4, 4),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(2, 2),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(3, 3),
                                               poolsize=(1, 1),
                                               activation=ReLU))

            l3out = self.layers[-1].outputDim
            self.layers.append(HiddenLayerParams(inputDim=(l3out[0], l3out[1] * l3out[2] * l3out[3]),
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(DropoutLayerParams(inputDim=self.layers[-1].outputDim,
                                                  outputDim=self.layers[-1].outputDim))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(DropoutLayerParams(inputDim=self.layers[-1].outputDim,
                                                  outputDim=self.layers[-1].outputDim))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 30),
                                                 activation=None))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, numJoints * nDims),
                                                 activation=None))

            self.outputDim = self.layers[-1].outputDim
        else:
            raise NotImplementedError("not implemented")


class PoseRegNet(NetBase):
    def __init__(self, rng, inputVar=None, cfgParams=None):
        """

        :type cfgParams: DescriptorNetParams
        """

        if cfgParams is None:
            raise Exception("Cannot create a Net without config parameters (ie. cfgParams==None)")

        if inputVar is None:
            inputVar = T.tensor4('x')  # input variable
        elif isinstance(inputVar, str):
            inputVar = T.tensor4(inputVar)  # input variable

        # create structure
        super(PoseRegNet, self).__init__(rng, inputVar, cfgParams)
