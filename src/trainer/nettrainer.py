"""Basis for different network trainers, manages memory.

NetTrainer provides interface for managing memory of different network trainer.
It should be derived by the individual trainer to enable a transparent usage.
NetTrainerParams is the parametrization of the NetTrainer.

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

import ctypes
import multiprocessing
import numpy
import time
import theano
import psutil
from data.importers import ICVLImporter
from net.convpoollayer import ConvPoolLayer
from util.handdetector import HandDetector

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class NetTrainerParams(object):
    def __init__(self):
        self.batch_size = 128
        self.momentum = 0.9
        self.learning_rate = 0.01
        self.weightreg_factor = 0.001  # regularization on the weights
        self.use_early_stopping = True
        self.lr_of_ep = lambda ep: self.learning_rate/(1+0.2*ep)  # learning rate as a function of epochs
        self.snapshot_freq = None  # make a intermediate save of the trained network all N epochs

class NetTrainer(object):
    """
    Basic class for different trainers that handels general memory management.
    Full training data (must be in RAM) is divided into chunks (macro batches) that are transferred to the GPU memory
    in blocks. Each macro batch consists of mini batches that are processed at once. If a mini batch is requested,
    which is not in the GPU memory, the macro block is automatically transferred.
    """

    def __init__(self, cfgParams, memory_factor):
        """
        Constructor
        :param cfgParams: initialized NetTrainerParams
        :param memory_factor: fraction of memory used for single shared variable
        """

        self.cfgParams = cfgParams

        if not isinstance(cfgParams, NetTrainerParams):
            raise ValueError("cfgParams must be an instance of NetTrainerParams")

        if 'gpu' in theano.config.device:
            # get GPU memory info
            mem_info = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
            self.memory = (mem_info[0] / 1024 ** 2) / float(memory_factor)  # MB, use third of free memory
        elif 'cpu' in theano.config.device:
            # get CPU memory info
            self.memory = (psutil.virtual_memory().available / 1024 ** 2) / float(memory_factor)  # MB, use third of free memory
        else:
            raise EnvironmentError("Neither GPU nor CPU device in theano.config found!")

        self.currentMacroBatch = -1  # current batch on GPU, load on first run
        self.trainSize = 0
        self.sampleSize = 0
        self.numTrainSamples = 0
        self.numValSamples = 0
        self.managedVar = []

    def addData(self, data):
        """
        Set the data of the network, not managed within training iterations, e.g. used for validation or other small data
        :param data: training data and labels specified as dictionary
        :return: None
        """

        if not isinstance(data, dict):
            raise ValueError("Error: expected dictionary for data!")

        for key in data:
            # no need to cache validation data
            setattr(self, key+'DB', self.alignData(data[key]))

            # shared variable already exists?
            if hasattr(self, key):
                print("Reusing shared variables!")
                getattr(self, key).set_value(getattr(self, key+'DB'), borrow=True)
            else:
                # create shared data
                setattr(self, key, theano.shared(getattr(self, key+'DB'), name=key, borrow=True))

    def addStaticData(self, data):
        """
        Set the data of the network, not managed within training iterations, e.g. used for validation or other small data
        :param data: training data and labels specified as dictionary
        :return: None
        """

        if not isinstance(data, dict):
            raise ValueError("Error: expected dictionary for data!")

        for key in data:
            # no need to cache validation data
            setattr(self, key+'DB', data[key])

            # shared variable already exists?
            if hasattr(self, key):
                print("Reusing shared variables!")
                getattr(self, key).set_value(getattr(self, key+'DB'), borrow=True)
            else:
                # create shared data
                setattr(self, key, theano.shared(getattr(self, key+'DB'), name=key, borrow=True))

    def addManagedData(self, data):
        """
        Set the data of the network, used for managing multiple input data sources for training
        :param data: training data and labels specified as dictionary
        :return: None
        """

        if not isinstance(data, dict):
            raise ValueError("Error: expected dictionary for data!")

        for key in data:
            # check sizes
            if (data[key].shape[0] != self.numTrainSamples):
                raise ValueError("Number of samples must be the same as number of labels.")

            if self.getNumMacroBatches() > 1:
                setattr(self, key+'DB', data[key][0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()])
                setattr(self, key+'DBlast', self.alignData(data[key][(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():]))
                self.managedVar.append(key)
            else:
                setattr(self, key+'DB', self.alignData(data[key]))

            # shared variable already exists?
            if hasattr(self, key):
                print("Reusing shared variables!")
                if self.trainSize > self.getGPUMemAligned():
                    print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                    # load first macro batch
                    idx = self.getNumSamplesPerMacroBatch()
                    self.replaceTrainingData(0, idx)
                else:
                    print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                    self.replaceTrainingData(0, self.train_data_xDB.shape[0])
            else:
                # load shared data
                if self.trainSize > self.getGPUMemAligned():
                    print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                    # load first macro batch
                    idx = self.getNumSamplesPerMacroBatch()
                    setattr(self, key, theano.shared(getattr(self, key+'DB')[:idx], name=key, borrow=True))
                else:
                    print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                    setattr(self, key, theano.shared(getattr(self, key+'DB'), name=key, borrow=True))

    def setData(self, train_data, train_y, val_data, val_y):
        """
        Set the data of the network, assuming train size << val size
        :param train_data: training data
        :param train_y: training labels
        :param val_data: validation data
        :param val_y: validation labels
        :return: None
        """

        # check sizes
        if (train_data.shape[0] != train_y.shape[0]) or (val_data.shape[0] != val_y.shape[0]):
            raise ValueError("Number of samples must be the same as number of labels.")

        # Check if the train_y is the image
        self.trainSize = max(train_data.nbytes, train_y.nbytes) / 1024. / 1024.
        self.numTrainSamples = train_data.shape[0]
        self.numValSamples = val_data.shape[0]
        self.sampleSize = self.trainSize / self.numTrainSamples

        # at least one minibatch per macro
        assert self.memory > self.sampleSize*self.cfgParams.batch_size

        # shrink macro batch size to smallest possible
        if self.getNumMacroBatches() == 1:
            self.memory = self.sampleSize * numpy.ceil(self.numTrainSamples/float(self.cfgParams.batch_size)) * self.cfgParams.batch_size

        # keep backup of original data
        # pad last macro batch separately to save memory
        if self.getNumMacroBatches() > 1:
            self.train_data_xDB = train_data[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
            self.train_data_xDBlast = self.alignData(train_data[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():])
            self.train_data_yDB = train_y[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
            self.train_data_yDBlast = self.alignData(train_y[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():])
            self.managedVar.append('train_data_x')
            self.managedVar.append('train_data_y')
        else:
            self.train_data_xDB = self.alignData(train_data)
            self.train_data_yDB = self.alignData(train_y)

        # no need to cache validation data
        self.val_data_xDB = val_data
        self.val_data_yDB = val_y

        print("Train size: {}MB, Memory available: {}MB, sample size: {}MB, aligned memory: {}MB".format(self.trainSize, self.memory,
                                                                                       self.sampleSize, self.getGPUMemAligned()))
        print("{} samples, batch size {}".format(train_data.shape[0], self.cfgParams.batch_size))
        print("{} macro batches, {} mini batches per macro, {} full mini batches total".format(self.getNumMacroBatches(),
                                                                                          self.getNumMiniBatchesPerMacroBatch(),
                                                                                          self.getNumMiniBatches()))

        # shared variable already exists?
        if hasattr(self, 'train_data_x'):
            print("Reusing shared variables!")
            if self.trainSize > self.getGPUMemAligned():
                print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                # load first macro batch
                idx = self.getNumSamplesPerMacroBatch()
                self.replaceTrainingData(0, idx)
                self.replaceValData(self.val_data_xDB, self.val_data_yDB)
            else:
                print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                self.replaceTrainingData(0, self.train_data_xDB.shape[0])
                self.replaceValData(self.val_data_xDB, self.val_data_yDB)
        else:
            # load shared data
            if self.trainSize > self.getGPUMemAligned():
                print("Loading {} macro batches a {}MB".format(self.getNumMacroBatches(), self.getGPUMemAligned()))
                # load first macro batch
                idx = self.getNumSamplesPerMacroBatch()
                self.train_data_x = theano.shared(self.train_data_xDB[:idx], name='train_data_x', borrow=True)
                self.train_data_y = theano.shared(self.train_data_yDB[:idx], name='train_data_y', borrow=True)
                self.val_data_x = theano.shared(self.val_data_xDB, name='val_data_x', borrow=True)
                self.val_data_y = theano.shared(self.val_data_yDB, name='val_data_y', borrow=True)
            else:
                print("Loading single macro batch {}/{}MB".format(self.trainSize, self.getGPUMemAligned()))
                self.train_data_x = theano.shared(self.train_data_xDB, name='train_data_x', borrow=True)
                self.train_data_y = theano.shared(self.train_data_yDB, name='train_data_y', borrow=True)
                self.val_data_x = theano.shared(self.val_data_xDB, name='val_data_x', borrow=True)
                self.val_data_y = theano.shared(self.val_data_yDB, name='val_data_y', borrow=True)

    def replaceTrainingData(self, start_idx, end_idx, last=False):
        """
        Replace the shared data of the training data
        :param start_idx: start index of data
        :param end_idx: end index of data
        :param last: specify if it is last macro-batch
        :return: None
        """

        for var in self.managedVar:
            if not hasattr(self, var):
                raise ValueError("Variable " + var + " not defined!")
            if last is True:
                getattr(self, var).set_value(getattr(self, var+'DBlast')[start_idx:end_idx], borrow=True)
            else:
                getattr(self, var).set_value(getattr(self, var+'DB')[start_idx:end_idx], borrow=True)

    def replaceValData(self, val_data, val_y):
        """
        Replace the shared data of the validation data, should not be necessary
        :param val_data: new validation data
        :param val_y: new validation labels
        :return: None
        """
        self.val_data_x.set_value(val_data, borrow=True)
        self.val_data_y.set_value(val_y, borrow=True)

    def alignData(self, data):
        """
        Align data to a multiple of the macro batch size, pad last incomplete minibatch with random samples
        :param data: data for alignment
        :return: padded data
        """
        # pad with zeros to macro batch size, but only along dimension 0 ie samples
        topad = self.getNumSamplesPerMacroBatch() - data.shape[0] % self.getNumSamplesPerMacroBatch()
        sz = []
        sz.append((0, topad))
        for i in range(len(data.shape) - 1):
            sz.append((0, 0))
        padded = numpy.pad(data, sz, mode='constant', constant_values=0)

        # fill last incomplete minibatch with random samples
        if (data.shape[0] % self.cfgParams.batch_size) != 0:
            # start from same random seed every time the data is padded, otherwise labels and data mix up
            rng = numpy.random.RandomState(data.shape[0])
            for i in xrange(0, self.cfgParams.batch_size - (data.shape[0] % self.cfgParams.batch_size)):
                padded[data.shape[0]+i] = padded[rng.randint(0, data.shape[0])]
        return padded

    def getSizeMiniBatch(self):
        """
        Get the size of a mini batch in MB
        :return: size of mini batch in MB
        """
        return self.cfgParams.batch_size * self.sampleSize

    def getSizeMacroBatch(self):
        """
        Get the size of a macro batch in MB
        :return: size of macro batch in MB
        """
        return self.getNumMacroBatches() * self.getSizeMiniBatch()

    def getNumFullMiniBatches(self):
        """
        Get total number of completely filled mini batches.
        WAS: drop last minibatch otherwise we might get problems with the zeropadding (all then zeros that are learnt)
        NOTE: as we augment last minibatch with random variables, we can use also last incomplete minibatch
        :return: number of training samples
        """
        # return int(numpy.floor(self.numTrainSamples / float(self.cfgParams.batch_size)))
        return self.getNumMiniBatches()

    def getNumMiniBatches(self):
        """
        Get total number of mini batches, including zero-padded patches
        :return: number of training samples
        """
        return int(numpy.ceil(self.numTrainSamples / float(self.cfgParams.batch_size)))

    def getNumMacroBatches(self):
        """
        Number of macro batches necessary for handling the training size
        :return: number of macro batches
        """
        return int(numpy.ceil(self.trainSize / float(self.getGPUMemAligned())))

    def getNumMiniBatchesPerMacroBatch(self):
        """
        Get number of mini batches per macro batch
        :return: number of mini batches per macro batch
        """
        return int(self.getGPUMemAligned() / self.sampleSize / self.cfgParams.batch_size)

    def getNumSamplesPerMacroBatch(self):
        """
        Get number of mini batches per macro batch
        :return: number of mini batches per macro batch
        """
        return int(self.getNumMiniBatchesPerMacroBatch() * self.cfgParams.batch_size)

    def getGPUMemAligned(self):
        """
        Get the number of MB of aligned GPU memory, aligned for full mini batches
        :return: usable size of GPU memory in MB
        """
        return self.sampleSize * self.cfgParams.batch_size * int(
            self.memory / float(self.sampleSize * self.cfgParams.batch_size))

    def loadMiniBatch(self, mini_idx):
        """
        Makes sure that the mini batch is loaded in the shared variable
        :param mini_idx: mini batch index
        :return: index within macro batch
        """
        macro_idx = int(mini_idx / self.getNumMiniBatchesPerMacroBatch())
        self.loadMacroBatch(macro_idx)
        return mini_idx % self.getNumMiniBatchesPerMacroBatch()

    def loadMacroBatch(self, macro_idx):
        """
        Make sure that macro batch is loaded in the shared variable
        :param macro_idx: macro batch index
        :return: None
        """
        if macro_idx != self.currentMacroBatch:
                # last macro batch is handled separately, as it is padded
                if self.isLastMacroBatch(macro_idx):
                    start_idx = 0
                    end_idx = self.getNumSamplesPerMacroBatch()
                    print("Loading last macro batch {}, start idx {}, end idx {}".format(macro_idx, start_idx, end_idx))
                    self.replaceTrainingData(start_idx, end_idx, last=True)
                    # remember current macro batch index
                    self.currentMacroBatch = macro_idx
                else:
                    start_idx = macro_idx * self.getNumSamplesPerMacroBatch()
                    end_idx = min((macro_idx + 1) * self.getNumSamplesPerMacroBatch(), self.train_data_xDB.shape[0])
                    print("Loading macro batch {}, start idx {}, end idx {}".format(macro_idx, start_idx, end_idx))
                    self.replaceTrainingData(start_idx, end_idx)
                    # remember current macro batch index
                    self.currentMacroBatch = macro_idx

    def isLastMacroBatch(self, macro_idx):
        """
        Check if macro batch is last macro batch
        :param macro_idx: macro batch index
        :return: True if batch is last macro batch
        """

        return macro_idx >= self.getNumMacroBatches()-1  # zero index

    def train(self, n_epochs=50, storeFilters=False):
        """
        Train the network using the self.train_model theano function
        :param n_epochs: number of epochs to train
        :param storeFilters: whether to store filters if ConvPoolLayer present
        :return: tuple (training costs, filter values, validation error) all over epochs
        """
        wvals = []

        # Use only full mini batches
        n_val_batches = self.val_data_xDB.shape[0] // self.cfgParams.batch_size

        # early-stopping parameters
        patience = n_epochs * self.getNumFullMiniBatches() / 2  # look as this many batches regardless (do at least half of the epochs we were asked for)
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(2*self.getNumFullMiniBatches(), patience / 2)
        # go through this many minibatches before checking the network on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        bestParams = None
        bestParamsEp = -1

        start_time = time.clock()

        train_costs = []
        validation_errors = []
        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            # save snapshot network
            if self.cfgParams.snapshot_freq is not None:
                if epoch % self.cfgParams.snapshot_freq == 0:
                    self.poseNet.save('net_{}'.format(epoch))

            epoch += 1
            # reduce learning rate
            learning_rate = self.cfgParams.lr_of_ep(epoch)
            for minibatch_index in xrange(self.getNumFullMiniBatches()):

                self.poseNet.enableDropout()

                # call parent to make sure batch is loaded
                self.epoch = epoch
                mini_idx = self.loadMiniBatch(minibatch_index)

                minibatch_avg_cost = self.train_model(mini_idx, learning_rate)

                if numpy.isnan(minibatch_avg_cost):
                    print("minibatch {0:4d}, average cost: NaN".format(minibatch_index))
                    # check which vars are nan
                    self.checkNaNs()

                    assert (False)

                print("minibatch {0:4d}, average cost: {1}".format(minibatch_index, minibatch_avg_cost))

                train_costs.append(minibatch_avg_cost)

                # iteration number ("we just did the iter_count-th batch")
                iter_count = (epoch - 1) * self.getNumFullMiniBatches() + minibatch_index

                if (iter_count + 1) % validation_frequency == 0:
                    # store filters
                    if storeFilters:
                        for lay in self.poseNet.layers:
                            if isinstance(lay, ConvPoolLayer):
                                if isinstance(lay.W, theano.compile.SharedVariable):
                                    wval = lay.W.get_value()
                                    wvals.append(wval)

                    self.poseNet.disableDropout()

                    # compute cost on validation set
                    validation_losses = [self.validation_cost(i) for i in xrange(n_val_batches)]
                    this_validation_loss = numpy.nanmean(validation_losses)

                    # compute error on validation set
                    validation_errors.append(numpy.nanmean([self.validation_error(i) for i in xrange(n_val_batches)]))

                    self.poseNet.enableDropout()

                    print('epoch %i, minibatch %i/%i, validation cost %f error %f' % \
                          (epoch, minibatch_index + 1, self.getNumFullMiniBatches(), this_validation_loss, validation_errors[-1]))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter_count * patience_increase)

                        best_validation_loss = this_validation_loss
                        # store best parameters
                        print("Best validation loss so far, store network weights!")
                        bestParams = self.poseNet.weightVals
                        bestParamsEp = epoch

        end_time = time.clock()
        print('Optimization complete with best validation score of %f,' % best_validation_loss)
        print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))

        # restore best params if given
        if bestParams is not None and self.cfgParams.use_early_stopping is True:
            self.poseNet.weightVals = bestParams
            print('Best params at epoch %d' % bestParamsEp)

        return (train_costs, wvals, validation_errors)

    def checkNaNs(self):
        """
        Check if there are NaN values in the weight or updates and print location
        :return: None
        """

        for param_i in self.params:
            if numpy.any(numpy.isnan(param_i.get_value())):
                print("NaN in weights")
