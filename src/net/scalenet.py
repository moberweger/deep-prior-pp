"""This file contains the bases classes for different networks

Created on 01.09.2014

@author: Markus Oberweger <oberweger@icg.tugraz.at>
"""
from __future__ import print_function

import theano
import theano.tensor as T
import time
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from net.dropoutlayer import DropoutLayer, DropoutLayerParams
from net.netbase import NetBase, NetBaseParams
import numpy
from util.helpers import ReLU


class ScaleNetParams(NetBaseParams):
    def __init__(self, type=0, nChan=1, wIn=128, hIn=128, batchSize=128, numJoints=16, nDims=3, resizeFactor = 2):
        '''
       Init the parametrization

        :type typeID: int
        :param typeID: type of descr network
        '''

        super(ScaleNetParams, self).__init__()

        self.batch_size = batchSize
        self.numJoints = numJoints
        self.nDims = nDims

        if type == 1:
            self.numInputs = 3
            self.inpConv = 3
            self.inputDim = [(batchSize, nChan, hIn, wIn), (batchSize, nChan, hIn//resizeFactor, wIn//resizeFactor), (batchSize, nChan, hIn//resizeFactor**2, wIn//resizeFactor**2)]
            # Try small configuration
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

            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, nChan, hIn//resizeFactor, wIn//resizeFactor),  # w,h,nChannel
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(2, 2),
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

            self.layers.append(ConvPoolLayerParams(inputDim=(batchSize, nChan, hIn//resizeFactor**2, wIn//resizeFactor**2),  # w,h,nChannel
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(2, 2),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(5, 5),
                                               poolsize=(1, 1),
                                               activation=ReLU))

            self.layers.append(ConvPoolLayerParams(inputDim=self.layers[-1].outputDim,
                                               nFilters=8,
                                               filterDim=(3, 3),
                                               poolsize=(1, 1),
                                               activation=ReLU))
            lout = 0
            for j in range(self.numInputs):
                idx = (j+1)*self.inpConv-1
                lout += self.layers[idx].outputDim[1]*self.layers[idx].outputDim[2]*self.layers[idx].outputDim[3]

            self.layers.append(HiddenLayerParams(inputDim=(batchSize, lout),
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, 1024),
                                                 activation=ReLU))

            self.layers.append(HiddenLayerParams(inputDim=self.layers[-1].outputDim,
                                                 outputDim=(batchSize, numJoints * nDims),
                                                 activation=None))  # last one is linear for regression

            self.outputDim = self.layers[-1].outputDim
        else:
            raise NotImplementedError("not implemented")


class ScaleNet(NetBase):
    def __init__(self, rng, inputVar=None, cfgParams=None, twin=None):
        '''

        :type cfgParams: DescriptorNetParams
        '''

        if cfgParams is None:
            raise Exception("Cannot create a Net without config parameters (ie. cfgParams==None)")

        if inputVar is None:
            self.inputVar = []
            for i in range(cfgParams.numInputs):
                self.inputVar.append(T.tensor4('x{}'.format(i)))
        else:
            raise Exception("Do not give inputVar, created inline")

        # create structure

        self.cfgParams = cfgParams

        # create network
        self.layers = []
        i = 0
        inI = 0
        for layerParam in cfgParams.layers:
            # first input is inputVar, otherwise input is output of last one
            if (i % self.cfgParams.inpConv) == 0 and i < self.cfgParams.numInputs*self.cfgParams.inpConv:
                inp = self.inputVar[inI]
                inI += 1
            else:
                # flatten output from conv to hidden layer
                if i == self.cfgParams.numInputs*self.cfgParams.inpConv:
                    cList = []
                    for j in range(self.cfgParams.numInputs):
                        idx = (j+1)*self.cfgParams.inpConv-1
                        cList.append(self.layers[idx].output.flatten(2))
                    inp = T.concatenate(cList, axis=1)
                else:
                    inp = self.layers[-1].output

            id = layerParam.__class__.__name__[:-6]
            constructor = globals()[id]
            self.layers.append(constructor(rng,
                                           inputVar=inp,
                                           cfgParams=layerParam,
                                           copyLayer=(None if (twin is None) else twin.layers[i]),
                                           layerNum=i))

            i += 1

        # assemble externally visible parameters
        self.output = self.layers[-1].output
