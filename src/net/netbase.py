"""Provides NetBase class for generating networks from configurations.

NetBase provides interface for building CNNs.
It should be inherited by all network classes in order to provide
basic functionality, ie computing outputs, creating computational
graph, managing dropout, etc.
NetBaseParams is the parametrization of these NetBase networks.

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

import difflib
import gzip
import time
import numpy
import cPickle
import re
from net.convpoollayer import ConvPoolLayer, ConvPoolLayerParams
from net.convlayer import ConvLayer, ConvLayerParams
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from net.poollayer import PoolLayer, PoolLayerParams
from net.dropoutlayer import DropoutLayer, DropoutLayerParams
from net.batchnormlayer import BatchNormLayer, BatchNormLayerParams
from net.nonlinearitylayer import NonlinearityLayer, NonlinearityLayerParams

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class NetBaseParams(object):
    def __init__(self):
        """
        Init the parametrization
        """

        self.numInputs = 1
        self.numOutputs = 1
        self.layers = []
        self.inputDim = None
        self.outputDim = None
        self.loadFile = None

    def getMemoryRequirement(self):
        """
        Get memory requirements of weights
        :return: memory requirement
        """
        mem = 0
        for l in self.layers:
            mem += l.getMemoryRequirement()
        return mem


class NetBase(object):
    def __init__(self, rng, inputVar, cfgParams, twin=None):
        """
        Initialize object by constructing the layers
        :param rng: random number generator
        :param inputVar: input variable
        :param cfgParams: parameters
        :param twin: determine to copy layer @deprecated
        :return: None
        """

        self._params_filter = []
        self._weights_filter = []

        self.inputVar = inputVar
        self.cfgParams = cfgParams
        self.rng = rng

        # create network
        self.layers = []
        i = 0
        for layerParam in cfgParams.layers:
            # first input is inputVar, otherwise input is output of last one
            if i == 0:
                inp = inputVar
            else:
                # flatten output from conv to hidden layer and reshape from hidden to conv layer
                if (len(self.layers[-1].cfgParams.outputDim) == 4) and (len(layerParam.inputDim) == 2):
                    inp = self.layers[-1].output.flatten(2)
                    inp.name = "input_layer_{}".format(i)  # name this node as it is different from previous output
                elif (len(layerParam.inputDim) == 4) and (len(self.layers[-1].cfgParams.outputDim) == 2):
                    inp = self.layers[-1].output.reshape(layerParam.inputDim, ndim=4)
                    inp.name = "input_layer_{}".format(i)  # name this node as it is different from previous output
                else:
                    inp = self.layers[-1].output

            id = layerParam.__class__.__name__[:-6]
            constructor = globals()[id]
            self.layers.append(constructor(rng,
                                           inputVar=inp,
                                           cfgParams=layerParam,
                                           copyLayer=None if (twin is None) else twin.layers[i],
                                           layerNum=i))

            i += 1

        # assemble externally visible parameters
        self.output = self.layers[-1].output
        
        self.load(self.cfgParams.loadFile)

    def __str__(self):
        """
        prints the parameters of the layers of the network
        :return: configuration string
        """

        cfg = "Network configuration:\n"
        i = 0
        for l in self.layers:
            cfg += "Layer {}: {} with {} \n".format(i, l.__class__.__name__, l)
            i += 1

        return cfg

    @property
    def all_params(self):
        """
        Get a list of all theano parameters for this network.
        :return: list of theano variables
        """
        prms = [p for l in self.layers for p in l.params]

        # only unique variables, remove shared weights from list
        return dict((obj.auto_name, obj) for obj in prms).values()

    @property
    def params(self):
        """
        Get a list of the learnable theano parameters for this network.
        :return: list of theano variables
        """

        # remove filtered params
        if not hasattr(self, '_params_filter'):
            self._params_filter = []
        prms = [p for l in self.layers for p in l.params if p.name not in self._params_filter]

        # only unique variables, remove shared weights from list
        return dict((obj.auto_name, obj) for obj in prms).values()

    @property
    def params_filter(self):
        return self._params_filter

    @params_filter.setter
    def params_filter(self, bl):
        names = [p.name for l in self.layers for p in l.params]
        for b in bl:
            if b not in names:
                raise UserWarning("Param {} not in model!".format(b))
        self._params_filter = bl

    @property
    def all_weights(self):
        """
        Get a list of all weights for this network.
        :return: list of theano variables
        """
        prms = [p for l in self.layers for p in l.weights]

        # only unique variables, remove shared weights from list
        return dict((obj.auto_name, obj) for obj in prms).values()

    @property
    def weights(self):
        """
        Get a list of the weights for this network.
        :return: list of theano variables
        """

        # remove filtered weights
        if not hasattr(self, '_weights_filter'):
            self._weights_filter = []
        prms = [p for l in self.layers for p in l.weights if p.name not in self._weights_filter]

        # only unique variables, remove shared weights from list
        return dict((obj.auto_name, obj) for obj in prms).values()

    @property
    def weights_filter(self):
        return self._weights_filter

    @weights_filter.setter
    def weights_filter(self, bl):
        names = [p.name for l in self.layers for p in l.weights]
        for b in bl:
            if b not in names:
                raise UserWarning("Weight {} not in model!".format(b))
        self._weights_filter = bl

    def computeOutput(self, inputs, timeit=False):
        """
        compute the output of the network for given input
        :param inputs: input data
        :param timeit: print the timing information
        :return: output of the network
        """
        import theano
        import theano.tensor as T

        # Convert input data
        if not isinstance(inputs, list):
            inputs = [inputs]

        # All data must be same
        assert all(i.shape[0] == inputs[0].shape[0] for i in inputs[1:])

        if not self.isDeterministic():
            print("WARNING: network is probabilistic for testing, DISABLING")
            self.setDeterministic()

        floatX = theano.config.floatX  # @UndefinedVariable
        batch_size = self.cfgParams.batch_size
        nSamp = inputs[0].shape[0]

        padSize = int(batch_size * numpy.ceil(nSamp / float(batch_size)))

        out = []
        if isinstance(self.output, list):
            for i in range(len(self.output)):
                outSize = list(self.cfgParams.outputDim[i])
                outSize[0] = padSize
                out.append(numpy.zeros(tuple(outSize), dtype=floatX))
        else:
            outSize = list(self.cfgParams.outputDim)
            outSize[0] = padSize
            out.append(numpy.zeros(tuple(outSize), dtype=floatX))

        index = T.lscalar('index')

        if not hasattr(self, 'compute_output'):
            self.input_data = []
            self.input_givens = dict()
            input_pad = []
            if inputs[0].shape[0] < batch_size:
                for k in range(len(inputs)):
                    shape = list(inputs[k].shape)
                    shape[0] = batch_size
                    input_pad.append(numpy.zeros(tuple(shape), dtype=inputs[k].dtype))
                    input_pad[k][0:inputs[k].shape[0]] = inputs[k][0:inputs[k].shape[0]]
                    input_pad[k][inputs[k].shape[0]:] = inputs[k][-1]
            else:
                for k in range(len(inputs)):
                    input_pad.append(inputs[k])
            for i in range(len(inputs)):
                if len(inputs) == 1 and not isinstance(self.inputVar, list):
                    self.input_data.append(theano.shared(input_pad[i][0:batch_size], self.inputVar.name, borrow=True))
                    self.input_givens[self.inputVar] = self.input_data[i][index * batch_size:(index + 1) * batch_size]
                else:
                    assert isinstance(self.inputVar, list)
                    self.input_data.append(theano.shared(input_pad[i][0:batch_size], self.inputVar[i].name, borrow=True))
                    self.input_givens[self.inputVar[i]] = self.input_data[i][index * batch_size:(index + 1) * batch_size]
            print("compiling compute_output() ...")
            self.compute_output = theano.function(inputs=[index], outputs=self.output, givens=self.input_givens,
                                                  mode='FAST_RUN', on_unused_input='warn')
            print("done")

        # iterate to save memory
        n_test_batches = padSize / batch_size
        start = time.time()
        for i in range(n_test_batches):
            # pad last batch to batch size
            if i == n_test_batches-1:
                input_pad = []
                for k in range(len(inputs)):
                    shape = list(inputs[k].shape)
                    shape[0] = batch_size
                    input_pad.append(numpy.zeros(tuple(shape), dtype=inputs[k].dtype))
                    input_pad[k][0:inputs[k].shape[0]-i*batch_size] = inputs[k][i*batch_size:]
                    input_pad[k][inputs[k].shape[0]-i*batch_size:] = inputs[k][-1]
                for k in range(len(inputs)):
                    self.input_data[k].set_value(input_pad[k], borrow=True)
            else:
                for k in range(len(inputs)):
                    self.input_data[k].set_value(inputs[k][i * batch_size:(i + 1) * batch_size], borrow=True)
            o = self.compute_output(0)
            if isinstance(self.output, list):
                for k in range(len(self.output)):
                    out[k][i * batch_size:(i + 1) * batch_size] = o[k]
            else:
                out[0][i * batch_size:(i + 1) * batch_size] = o.reshape(self.cfgParams.outputDim)
        end = time.time()
        if timeit:
            print("{} in {}s, {}ms per frame".format(padSize, end - start, (end - start)*1000./padSize))
        if isinstance(self.output, list):
            for k in range(len(self.output)):
                out[k] = out[k][0:nSamp]
            return out
        else:
            return out[0][0:nSamp]

    def unsetDeterministic(self):
        """
        Enables dropout and batch normalization in all layers, ie for training
        :return: None
        """
        for layer in self.layers:
            if isinstance(layer, DropoutLayer) or isinstance(layer, BatchNormLayer):
                layer.unsetDeterministic()

    def setDeterministic(self):
        """
        Disables dropout and batch normalizatoin in all layers, ie for testing
        :return: None
        """
        for layer in self.layers:
            if isinstance(layer, DropoutLayer) or isinstance(layer, BatchNormLayer):
                layer.setDeterministic()

    def isDeterministic(self):
        """
        Checks if forward pass in network is deterministic
        :return: None
        """
        for layer in self.layers:
            if isinstance(layer, DropoutLayer) or isinstance(layer, BatchNormLayer):
                if not layer.isDeterministic():
                    return False

        return True

    def hasDropout(self):
        """
        Checks if network has dropout layers
        :return: True if there are dropout layers
        """
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                return True

        return False

    @property
    def weightVals(self):
        """
        Returns list of the weight values
        :return: list of weight values
        """
        return self.recGetWeightVals(self.all_params)

    @weightVals.setter
    def weightVals(self, value):
        """
        Set weights with given values
        :param value: values for weights
        :return: None
        """
        self.recSetWeightVals(self.all_params, value)

    def recSetWeightVals(self, param, value):
        """
        Set weights with given values
        :param param: layer parameters listing the layers weights
        :param value: values for weights
        :return: None
        """
        if isinstance(value, list):
            assert isinstance(param, list), "tried to assign a list of weights to params, which is not a list {}".format(type(param))
            assert len(param) == len(value), "tried to assign unequal list of weights {} != {}".format(len(param), len(value))
            for i in xrange(len(value)):
                self.recSetWeightVals(param[i], value[i])
        else:
            param.set_value(value)

    def recGetWeightVals(self, param):
        """
        Returns list of the weight values
        :param param: layer parameters listing the layers weights
        :return: list of weight values
        """
        w = []
        if isinstance(param, list):
            for p in param:
                w.append(self.recGetWeightVals(p))
        else:
            w = param.get_value()
        return w

    def save(self, filename):
        """
        Save the state of this network to a pickle file on disk.
        :param filename: Save the parameters of this network to a pickle file at the named path. If this name ends in
               ".gz" then the output will automatically be gzipped; otherwise the output will be a "raw" pickle.
        :return: None
        """

        state = dict([('class', self.__class__.__name__), ('network', self.__str__())])
        for layer in self.layers:
            key = '{}-values'.format(layer.layerNum)
            state[key] = [p.get_value() for p in layer.params]
            state[key].extend([p.get_value() for p in layer.params_nontrained])
        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'wb')
        cPickle.dump(state, handle, -1)
        handle.close()
        print 'Saved model parameter to {}'.format(filename)

    def load(self, filename):
        """
        Load the parameters for this network from disk.
        :param filename: Load the parameters of this network from a pickle file at the named path. If this name ends in
               ".gz" then the input will automatically be gunzipped; otherwise the input will be treated as a "raw" pickle.
        :return: None
        """
        if filename is None:
            return

        opener = gzip.open if filename.lower().endswith('.gz') else open
        handle = opener(filename, 'rb')
        saved = cPickle.load(handle)
        handle.close()
        if saved['network'] != self.__str__():
            print "Possibly not matching network configuration!"
            differences = list(difflib.Differ().compare(saved['network'].splitlines(), self.__str__().splitlines()))
            print "Differences are:"
            print "\n".join(differences)
        for layer in self.layers:
            if (len(layer.params) + len(layer.params_nontrained)) != len(saved['{}-values'.format(layer.layerNum)]):
                print "Warning: Layer parameters for layer {} do not match. Trying to fit on shape!".format(layer.layerNum)
                n_assigned = 0
                for p in layer.params + layer.params_nontrained:
                    for v in saved['{}-values'.format(layer.layerNum)]:
                        if p.get_value().shape == v.shape:
                            p.set_value(v)
                            n_assigned += 1

                if n_assigned != (len(layer.params) + len(layer.params_nontrained)):
                    raise ImportError("Could not load all necessary variables!")
                else:
                    print "Found fitting parameters!"
            else:
                for p, v in zip(layer.params + layer.params_nontrained, saved['{}-values'.format(layer.layerNum)]):
                    if p.get_value().shape == v.shape:
                        p.set_value(v)
                    else:
                        print "WARNING: Skipping parameter for {}! Shape {} does not fit {}.".format(p.name, p.get_value().shape, v.shape)
        print 'Loaded model parameters from {}'.format(filename)
