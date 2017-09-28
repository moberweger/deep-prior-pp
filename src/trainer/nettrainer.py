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

import multiprocessing
import numpy
import time
import gc
import sharedmem
import theano
import psutil
from net.convlayer import ConvLayer
from net.convpoollayer import ConvPoolLayer
from util.handdetector import HandDetector
from util.helpers import chunks

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
        self.lr_of_ep = lambda ep: numpy.float32(self.learning_rate/10.) if ep <= 1 else numpy.float32(self.learning_rate*numpy.exp(-0.04*ep))  # learning rate as a function of epochs
        self.snapshot_last = 5  # make an backup copy every 5 epochs
        self.snapshot_freq = None  # make an intermediate save of the trained network all N epochs
        # parallel augmentation, off by default
        self.para_augment = False
        self.para_num_proc = 8
        self.augment_fun_params = {'fun': None, 'args': {}}
        # parallel loading, off by default
        self.para_load = False
        self.load_fun_params = {'fun': None, 'args': {}}
        # force to reload each macrobatch, ie call replaceTrainingData, also if macrobatch already loaded
        self.force_macrobatch_reload = False
        # pad with random data or by repeating last sample
        self.pad_random = True
        self.validation_frequency = 1000  # run validation all N minibatches
        self.pre_epoch_fn = None
        self.post_epoch_fn = None
        self.pre_minibatch_fn = None
        self.post_minibatch_fn = None


class NetTrainer(object):
    """
    Basic class for different trainers that handels general memory management.
    Full training data (must be in RAM) is divided into chunks (macro batches) that are transferred to the GPU memory
    in blocks. Each macro batch consists of mini batches that are processed at once. If a mini batch is requested,
    which is not in the GPU memory, the macro block is automatically transferred.
    """

    SYNC_BATCH_FINISHED = 'batch_finished'
    SYNC_LOAD_FINISHED = 'load_finished'

    def __init__(self, cfgParams, memory_factor, subfolder='./eval/', numChunks=1):
        """
        Constructor
        :param cfgParams: initialized NetTrainerParams
        :param memory_factor: fraction of memory used for single shared variable
        """

        self.subfolder = subfolder
        self.cfgParams = cfgParams
        self.rng = numpy.random.RandomState(23455)

        if not isinstance(cfgParams, NetTrainerParams):
            raise ValueError("cfgParams must be an instance of NetTrainerParams")

        # use fraction of free memory
        if 'gpu' in theano.config.device:
            # get GPU memory info
            mem_info = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
            self.memorySize = (mem_info[0] / 1024 ** 2) / float(memory_factor)  # MB
        elif 'cpu' in theano.config.device:
            # get CPU memory info
            self.memorySize = (psutil.virtual_memory().available / 1024 ** 2) / float(memory_factor)  # MB
        else:
            raise EnvironmentError("Neither GPU nor CPU device in theano.config found!")

        if cfgParams.para_load is True and numChunks == 1:
            raise ValueError("para_load is True but numChunks == 1, so we do not need para_load!")

        self.currentMacroBatch = -1  # current batch on GPU, load on first run
        self.currentChunk = -1  # current chunk in RAM, load on first run
        self.numChunks = numChunks
        self.trainSize = 0
        self.sampleSize = 0
        self.numTrainSamplesMB = 0
        self.numTrainSamples = 0
        self.numValSamples = 0
        self.epoch = 0
        self.managedVar = []
        self.trainingVar = []
        self.validation_observer = []

    def addData(self, data):
        """
        Set the data of the network, not managed within training iterations, 
        e.g. used for validation or other small data
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
        Set the data of the network, not managed within training iterations, 
        e.g. used for validation or other small data
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
            if data[key].shape[0] != self.numTrainSamplesMB:
                raise ValueError("Number of samples must be the same as number of labels.")

            if self.getNumMacroBatches() > 1:
                setattr(self, key+'DB', data[key][0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()])
                setattr(self, key+'DBlast', self.alignData(data[key][(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():], fillData=data[key]))
                self.managedVar.append(key)
            else:
                setattr(self, key+'DB', self.alignData(data[key]))
            self.trainingVar.append(key)

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

    def setData(self, train_data, train_y, val_data, val_y, max_train_size=0):
        """
        Set the data of the network, assuming train size << val size
        :param train_data: training data
        :param train_y: training labels
        :param val_data: validation data
        :param val_y: validation labels
        :param max_train_size: optional if training data has additional large chunk
        :return: None
        """

        # check sizes
        if (train_data.shape[0] != train_y.shape[0]) or (val_data.shape[0] != val_y.shape[0]):
            raise ValueError("Number of samples must be the same as number of labels.")

        # Check if the train_y is the image
        self.trainSize = max(train_data.nbytes, train_y.nbytes, max_train_size) / 1024. / 1024.
        self.numTrainSamplesMB = train_data.shape[0]
        self.numTrainSamples = self.numTrainSamplesMB
        self.numValSamples = val_data.shape[0]
        self.sampleSize = self.trainSize / self.numTrainSamplesMB

        # at least one minibatch per macro
        assert self.memorySize > self.sampleSize*self.cfgParams.batch_size, "{} > {}".format(self.memorySize, self.sampleSize*self.cfgParams.batch_size)

        # shrink macro batch size to smallest possible
        if self.getNumMacroBatches() == 1:
            self.memorySize = self.sampleSize * numpy.ceil(self.numTrainSamplesMB/float(self.cfgParams.batch_size)) * self.cfgParams.batch_size

        # keep backup of original data
        # pad last macro batch separately to save memory
        if self.getNumMacroBatches() > 1:
            self.train_data_xDB = train_data[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
            self.train_data_xDBlast = self.alignData(train_data[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():], fillData=train_data)
            self.train_data_yDB = train_y[0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
            self.train_data_yDBlast = self.alignData(train_y[(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():], fillData=train_y)
            self.managedVar.append('train_data_x')
            self.managedVar.append('train_data_y')
        else:
            self.train_data_xDB = self.alignData(train_data)
            self.train_data_yDB = self.alignData(train_y)

        self.trainingVar.append('train_data_x')
        self.trainingVar.append('train_data_y')

        # no need to cache validation data
        self.val_data_xDB = val_data
        self.val_data_yDB = val_y

        print("Train size: {}MB, Memory available: {}MB, sample size: {}MB, aligned memory: {}MB".format(
            self.trainSize, self.memorySize, self.sampleSize, self.getGPUMemAligned()))
        print("{} train samples, {} val samples, batch size {}".format(
            train_data.shape[0], val_data.shape[0], self.cfgParams.batch_size))
        print("{} macro batches, {} mini batches per macro, {} full mini batches total".format(
            self.getNumMacroBatches(), self.getNumMiniBatchesPerMacroBatch(), self.getNumMiniBatches()))
        print("{} data chunks, {} train samples total".format(self.numChunks, self.numTrainSamples))

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

    def replaceTrainingData(self, start_idx, end_idx=None, last=False):
        """
        Replace the shared data of the training data
        :param start_idx: start index of data
        :param end_idx: end index of data
        :param last: specify if it is last macro-batch
        :return: None
        """

        if end_idx is None:
            assert isinstance(start_idx, list) or isinstance(start_idx, numpy.ndarray)

        for var in self.managedVar:
            if not hasattr(self, var):
                raise ValueError("Variable " + var + " not defined!")
            if last is True:
                if end_idx is None:
                    getattr(self, var).set_value(getattr(self, var+'DBlast')[start_idx], borrow=True)
                else:
                    getattr(self, var).set_value(getattr(self, var+'DBlast')[start_idx:end_idx], borrow=True)
            else:
                if end_idx is None:
                    getattr(self, var).set_value(getattr(self, var+'DB')[start_idx], borrow=True)
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

    def alignData(self, data, alignSize=None, out=None, fillData=None):
        """
        Align data to a multiple of the macro batch size, pad last incomplete minibatch with random samples
        :param data: data for alignment
        :param out: reuse data storage for padded data
        :param fillData: data used for padding last minibatches
        :return: padded data
        """

        if alignSize is None:
            alignSize = self.getNumSamplesPerMacroBatch()

        if alignSize < data.shape[0]:
            print "WARNING: aligned size < data size ({}<{})".format(alignSize, data.shape[0])

        # pad with zeros to macro batch size, but only along dimension 0 ie samples
        if data.shape[0] == alignSize:
            topad = 0
        else:
            topad = alignSize - data.shape[0] % alignSize
        sz = [(0, topad)]
        for i in range(len(data.shape) - 1):
            sz.append((0, 0))
        if out is not None:
            raise NotImplementedError()
            assert out.shape[0] == alignSize
            assert all(out.shape[1:] == data.shape[1:])
            padded = out
            padded[0:data.shape[0]] = data
        else:
            padded = numpy.pad(data, sz, mode='constant', constant_values=0)

        if fillData is None:
            fillData = padded

        # fill full macrobatch with random data, just to be sure...
        if (data.shape[0] % alignSize) != 0:
            # fill last incomplete minibatch with random samples or by repeating last one
            if self.cfgParams.pad_random:
                # start from same random seed every time the data is padded, otherwise labels and data mix up
                rng = numpy.random.RandomState(data.shape[0])
                for i in xrange(0, alignSize - (data.shape[0] % alignSize)):
                    padded[data.shape[0]+i] = fillData[rng.randint(0, fillData.shape[0])]
            else:
                for i in xrange(0, alignSize - (data.shape[0] % alignSize)):
                    padded[data.shape[0]+i] = padded[data.shape[0]-1]

        if out is None:
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

    def getNumMiniBatchesPerChunk(self):
        """
        Get number of mini batches per macro batch
        :return: number of mini batches per macro batch
        """
        return int(self.getNumMiniBatchesPerMacroBatch() * self.getNumMacroBatches())

    def getGPUMemAligned(self):
        """
        Get the number of MB of aligned GPU memory, aligned for full mini batches
        :return: usable size of GPU memory in MB
        """
        return self.sampleSize * self.cfgParams.batch_size * int(
            self.memorySize / float(self.sampleSize * self.cfgParams.batch_size))

    def loadMiniBatch(self, mini_idx):
        """
        Makes sure that the mini batch is loaded in the shared variable
        :param mini_idx: mini batch index
        :return: index within macro batch
        """
        macro_idx = int((mini_idx % self.getNumMiniBatchesPerChunk()) / self.getNumMiniBatchesPerMacroBatch())
        self.loadMacroBatch(macro_idx, mini_idx)
        return mini_idx % self.getNumMiniBatchesPerMacroBatch()

    def loadMacroBatch(self, macro_idx, mini_idx):
        """
        Make sure that macro batch is loaded in the shared variable
        :param macro_idx: macro batch index
        :param mini_idx: mini batch index
        :return: None
        """

        def do_para_swap():
            if self.cfgParams.para_load is True:
                if self.isLastMacroBatch(macro_idx):
                    # copy data to RAM when ready
                    (ci, msg) = self.load_send_queue.get()
                    assert msg == self.SYNC_LOAD_FINISHED

                    for var in self.trainingVar:
                        if not hasattr(self, var):
                            raise ValueError("Variable " + var + " not defined!")
                        if self.getNumMacroBatches() > 1:
                            getattr(self, var+'DB')[:] = self.load_data_queue[var][0:(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch()]
                            getattr(self, var+'DBlast')[:] = self.alignData(self.load_data_queue[var][(self.getNumMacroBatches()-1)*self.getNumSamplesPerMacroBatch():], fillData=self.load_data_queue[var])
                        else:
                            getattr(self, var+'DB')[:] = self.load_data_queue[var]
                    self.currentChunk = ci
                    next_chunk = numpy.mod(ci + 1, self.numChunks)
                    print("Received chunk {}, requesting {}".format(ci, next_chunk))
                    print("Loading chunk {}, last {}".format(next_chunk, False))
                    self.load_recv_queue.put((next_chunk, self.cfgParams.load_fun_params, False))

        force_reload = (((mini_idx % self.getNumMiniBatchesPerChunk()) == self.getNumMiniBatchesPerMacroBatch()-1) and self.cfgParams.force_macrobatch_reload is True and (self.getNumMacroBatches() == 1))
        if macro_idx != self.currentMacroBatch or force_reload is True:
            if self.cfgParams.para_augment is True:
                # copy data to GPU when ready
                old_mbi = -1
                mbi = -1
                for s in self.augment_send_queue:
                    (mbi, msg) = s.get()
                    assert msg == self.SYNC_BATCH_FINISHED
                    if old_mbi == -1:
                        old_mbi = mbi
                    else:
                        assert old_mbi == mbi

                new_data = self.augment_data_queue
                for var in self.trainingVar:
                    if not hasattr(self, var):
                        raise ValueError("Variable " + var + " not defined!")
                    # No borrow, since we modify the underlying memory
                    getattr(self, var).set_value(new_data[var], borrow=False)
                self.currentMacroBatch = mbi

                # swap data before we start augmenting new one
                do_para_swap()

                next_mbi = numpy.mod(mbi + 1, self.getNumMacroBatches())
                print("Received macro batch {}, requesting {}".format(mbi, next_mbi))
                last, tidx, idxs = self.chunksForMP(next_mbi)
                print("Loading macro batch {}, last {}, start idx {}, end idx {}".format(next_mbi, last, numpy.min(idxs), numpy.max(idxs)))
                for i, r in enumerate(self.augment_recv_queue):
                    r.put((next_mbi, self.cfgParams.augment_fun_params, last, tidx[i], idxs[i]))
            elif self.cfgParams.augment_fun_params['fun'] is not None:
                # singe thread augmentation
                new_data = self.augment_data_queue

                # invoke function to generate new data
                last, tidx, idxs = self.chunksForMP(macro_idx)
                print("Loading macro batch {}, last {}, start idx {}, end idx {}".format(macro_idx, last, numpy.min(idxs), numpy.max(idxs)))
                getattr(self, self.cfgParams.augment_fun_params['fun'])(self.cfgParams.augment_fun_params, macro_idx,
                                                                        last,
                                                                        [itm for sl in tidx for itm in sl],
                                                                        [itm for sl in idxs for itm in sl],
                                                                        new_data)
                for var in self.trainingVar:
                    if not hasattr(self, var):
                        raise ValueError("Variable " + var + " not defined!")
                    # No borrow, since we modify the underlying memory
                    getattr(self, var).set_value(new_data[var], borrow=False)
                # remember current macro batch index
                self.currentMacroBatch = macro_idx

                # swap data
                do_para_swap()
            else:
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

                # swap data
                do_para_swap()

    def loadMacroBatchMP(self, recv_queue, send_queue, data_queue):
        """
        Function which is started as thread, that loads and prepares macro batches.
        It can further be used to load augmented data from the macro batches, with doing the augmentation on CPU,
        and the calculations in parallel on the GPU.
        :param recv_queue: recv_queue is only for receiving
        :param send_queue: send_queue is only for sending
        :return: None
        """

        self.rng.seed()  # get new seed for each process

        while True:
            # getting the macro batch index to load
            (macro_idx, params, last, tidxs, idxs) = recv_queue.get()

            # kill signal
            if macro_idx == -1:
                return

            # invoke function to generate new data
            assert params['fun'] is not None
            assert len(idxs) > 0
            assert len(idxs) == len(tidxs), "{} != {}".format(len(idxs), len(tidxs))
            getattr(self, params['fun'])(params, macro_idx, last, tidxs, idxs, data_queue)

            # put new data ready
            send_queue.put((macro_idx, self.SYNC_BATCH_FINISHED))

    def loadDataMP(self, recv_queue, send_queue, data_queue):
        """
        Function which is started as thread, that loads and prepares macro batches.
        It can further be used to load augmented data from the macro batches, with doing the augmentation on CPU,
        and the calculations in parallel on the GPU.
        :param recv_queue: recv_queue is only for receiving
        :param send_queue: send_queue is only for sending
        :return: None
        """

        self.rng.seed()  # get new seed for each process

        while True:
            # getting the macro batch index to load
            (chunk_idx, params, last) = recv_queue.get()

            # kill signal
            if chunk_idx == -1:
                return

            # invoke function to generate new data
            assert params['fun'] is not None
            getattr(self, params['fun'])(params, chunk_idx, last, data_queue)

            # put new data ready
            send_queue.put((chunk_idx, self.SYNC_LOAD_FINISHED))

    def isLastMacroBatch(self, macro_idx):
        """
        Check if macro batch is last macro batch
        :param macro_idx: macro batch index
        :return: True if batch is last macro batch
        """

        return macro_idx >= self.getNumMacroBatches()-1  # zero index

    def setupDataLoading(self):
        if self.cfgParams.para_augment is True:
            # create communication queues
            self.augment_recv_queue = [multiprocessing.Queue() for _ in range(self.cfgParams.para_num_proc)]
            self.augment_send_queue = [multiprocessing.Queue() for _ in range(self.cfgParams.para_num_proc)]
            self.augment_data_queue = dict()
            for var in self.trainingVar:
                if not hasattr(self, var):
                    raise ValueError("Variable " + var + " not defined!")
                if var.startswith("train_"):
                    self.augment_data_queue[var] = sharedmem.empty_like(getattr(self, var+'DB')[0:self.getNumSamplesPerMacroBatch()])

            # start loading process (no threads because of GIL)
            self.augment_processes = [multiprocessing.Process(target=self.loadMacroBatchMP,
                                                              args=(self.augment_recv_queue[i], self.augment_send_queue[i], self.augment_data_queue)) for i in range(self.cfgParams.para_num_proc)]
            for p in self.augment_processes:
                p.start()

            # prepare next one
            last, tidx, idxs = self.chunksForMP(0)
            print("Loading macro batch {}, last {}, start idx {}, end idx {}".format(0, last, numpy.min(idxs), numpy.max(idxs)))
            for i, r in enumerate(self.augment_recv_queue):
                r.put((0, self.cfgParams.augment_fun_params, last, tidx[i], idxs[i]))
        elif self.cfgParams.augment_fun_params['fun'] is not None:
            self.augment_data_queue = dict()
            for var in self.trainingVar:
                if not hasattr(self, var):
                    raise ValueError("Variable " + var + " not defined!")
                if var.startswith("train_"):
                    self.augment_data_queue[var] = numpy.zeros_like(getattr(self, var + 'DB')[0:self.getNumSamplesPerMacroBatch()])
        else:
            pass

        if self.cfgParams.para_load is True:
            assert self.numChunks > 1, "Please set the number of chunks appropriately!"
            self.load_recv_queue = multiprocessing.Queue()
            self.load_send_queue = multiprocessing.Queue()
            self.load_data_queue = dict()
            for var in self.trainingVar:
                if not hasattr(self, var):
                    raise ValueError("Variable " + var + " not defined!")
                if var.startswith("train_"):
                    sz = list(getattr(self, var+'DB').shape)
                    if hasattr(self, var+'DBlast'):
                        sz[0] += getattr(self, var+'DBlast').shape[0]
                    self.load_data_queue[var] = sharedmem.empty(tuple(sz), dtype=getattr(self, var+'DB').dtype)

            # start loading process (no threads because of GIL)
            self.load_process = multiprocessing.Process(target=self.loadDataMP,
                                                        args=(self.load_recv_queue, self.load_send_queue, self.load_data_queue))
            self.load_process.start()

            # prepare next one
            print("Loading chunk {}, last {}".format(0, False))
            self.load_recv_queue.put((0, self.cfgParams.load_fun_params, False))
        else:
            pass

    def chunksForMP(self, mbi, use_all_last=True):
        if self.isLastMacroBatch(mbi):
            start_idx = 0
            if use_all_last is True:
                end_idx = self.getNumSamplesPerMacroBatch()
            else:
                num_mb = int(numpy.ceil(self.numTrainSamplesMB / float(self.cfgParams.batch_size)))
                end_idx = self.cfgParams.batch_size * (num_mb - self.getNumMiniBatchesPerMacroBatch() * (self.getNumMacroBatches() - 1))
            last = True
        else:
            start_idx = mbi * self.getNumSamplesPerMacroBatch()
            end_idx = min((mbi + 1) * self.getNumSamplesPerMacroBatch(), self.train_data_xDB.shape[0])
            last = False

        num_chunks = int(numpy.ceil((end_idx - start_idx) / float(self.cfgParams.para_num_proc)))
        idxs = list(chunks(range(start_idx, end_idx), num_chunks))
        tidxs = list(chunks(range(0, (end_idx - start_idx)), num_chunks))
        return last, tidxs, idxs

    def unsetDataLoading(self):
        if self.cfgParams.para_augment is True:
            # send the thread finish code
            for r in self.augment_recv_queue:
                r.put((-1, self.cfgParams.augment_fun_params, 0, [], []))
            # empty queues and join
            for q in self.augment_send_queue:
                q.get()
            for p in self.augment_processes:
                p.join()
            delattr(self, 'augment_recv_queue')
            delattr(self, 'augment_send_queue')
            delattr(self, 'augment_data_queue')
            delattr(self, 'augment_processes')
        elif self.cfgParams.augment_fun_params['fun'] is not None:
            delattr(self, 'augment_data_queue')
        else:
            pass

        if self.cfgParams.para_load is True:
            # send the thread finish code
            self.load_recv_queue.put((-1, self.cfgParams.load_fun_params, False))
            # empty queues and join
            self.load_send_queue.get()
            self.load_process.join()
            delattr(self, 'load_recv_queue')
            delattr(self, 'load_send_queue')
            delattr(self, 'load_data_queue')
            delattr(self, 'load_process')
        else:
            pass

    def train(self, n_epochs=50, storeFilters=False):
        """
        Train the network using the self.train_model theano function
        :param n_epochs: number of epochs to train
        :param storeFilters: whether to store filters if ConvPoolLayer present
        :return: tuple (training costs, filter values, validation error) all over epochs
        """

        if len(self.validation_observer) < 1:
            raise ValueError("Require at least 1 validation function, that monitors validation cost!")

        # parallel loading of macro batches
        if self.cfgParams.augment_fun_params['fun'] is not None or self.cfgParams.load_fun_params['fun'] is not None:
            self.setupDataLoading()

        wvals = []

        # Use only full mini batches
        n_val_batches = self.val_data_xDB.shape[0] // self.cfgParams.batch_size

        best_validation_loss = numpy.inf
        bestParams = None
        bestParamsEp = -1

        start_time = time.clock()

        train_costs = []
        validation_obs = [[] for x in xrange(1, len(self.validation_observer))]
        self.epoch = 0

        self.poseNet.setDeterministic()
        # compute observers on validation set
        for vi in range(1, len(self.validation_observer)):
            validation_obs[vi-1].append(numpy.nanmean([self.validation_observer[vi](i) for i in xrange(n_val_batches)]))
        self.poseNet.unsetDeterministic()

        while self.epoch < n_epochs:
            # save snapshot network
            if self.epoch % self.cfgParams.snapshot_last == 0:
                self.poseNet.save(self.subfolder+'/net_last.pkl')
            if self.cfgParams.snapshot_freq is not None:
                if self.epoch % self.cfgParams.snapshot_freq == 0:
                    self.poseNet.save(self.subfolder+'/net_{}.pkl'.format(self.epoch))

            if self.cfgParams.pre_epoch_fn is not None:
                # invoke function
                getattr(self, self.cfgParams.pre_epoch_fn)()

            self.epoch += 1
            # reduce learning rate
            learning_rate = self.cfgParams.lr_of_ep(self.epoch)
            for minibatch_index in xrange(self.getNumFullMiniBatches()):

                if self.cfgParams.pre_minibatch_fn is not None:
                    # invoke function
                    getattr(self, self.cfgParams.pre_minibatch_fn)()

                self.poseNet.unsetDeterministic()

                # call parent to make sure batch is loaded
                mini_idx = self.loadMiniBatch(minibatch_index)

                minibatch_avg_cost = self.train_model(mini_idx, learning_rate)

                if numpy.any(numpy.isnan(minibatch_avg_cost)):
                    print("minibatch {0:4d}, average cost: NaN".format(minibatch_index))
                    # check which vars are nan
                    self.checkNaNs()

                    assert False

                print("minibatch {0:4d}, average cost: {1}".format(minibatch_index, minibatch_avg_cost))

                train_costs.append(minibatch_avg_cost)

                if self.cfgParams.post_minibatch_fn is not None:
                    # invoke function
                    getattr(self, self.cfgParams.post_minibatch_fn)()

                # iteration number ("we just did the iter_count-th batch")
                iter_count = (self.epoch - 1) * self.getNumFullMiniBatches() + minibatch_index

                if (iter_count + 1) % self.cfgParams.validation_frequency == 0:
                    # store filters
                    if storeFilters:
                        for lay in self.poseNet.layers:
                            if isinstance(lay, ConvPoolLayer) or isinstance(lay, ConvLayer):
                                if isinstance(lay.W, theano.compile.SharedVariable):
                                    wval = lay.W.get_value()
                                    wvals.append(wval)

                    self.poseNet.setDeterministic()

                    # compute cost on validation set
                    this_validation_loss = numpy.nanmean([self.validation_observer[0](i) for i in xrange(n_val_batches)])

                    # compute observers on validation set
                    for vi in range(1, len(self.validation_observer)):
                        validation_obs[vi-1].append(numpy.nanmean([self.validation_observer[vi](i) for i in xrange(n_val_batches)]))

                    self.poseNet.unsetDeterministic()

                    print "{}: epoch {}, minibatch {}/{}, validation cost {} error {}".format(time.ctime(),
                                                                                              self.epoch,
                                                                                              minibatch_index + 1,
                                                                                              self.getNumFullMiniBatches(),
                                                                                              this_validation_loss,
                                                                                              [vo[-1] for vo in validation_obs])

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        best_validation_loss = this_validation_loss
                        # store best parameters
                        print("Best validation loss so far, store network weights!")
                        bestParams = self.poseNet.weightVals
                        bestParamsEp = self.epoch

            if self.cfgParams.post_epoch_fn is not None:
                # invoke function
                getattr(self, self.cfgParams.post_epoch_fn)()

        end_time = time.clock()
        print('Optimization complete with best validation score of %f,' % best_validation_loss)
        print('The code run for %d epochs, with %f epochs/sec' % (self.epoch, self.epoch / (end_time - start_time)))

        # restore best params if given
        if bestParams is not None and self.cfgParams.use_early_stopping is True:
            self.poseNet.weightVals = bestParams
            print('Best params at epoch %d' % bestParamsEp)

        if self.cfgParams.augment_fun_params['fun'] is not None:
            self.unsetDataLoading()

        return train_costs, wvals, validation_obs[0] if len(validation_obs) == 1 else validation_obs

    def checkNaNs(self):
        """
        Check if there are NaN values in the weight or updates and print location
        :return: None
        """

        for param_i in self.params:
            if numpy.any(numpy.isnan(param_i.get_value())):
                print("NaN in weights", param_i.name)

    def augmentCrop(self, img, gt3Dcrop, com, cube, M, aug_modes, hd, normZeroOne=False, sigma_com=None,
                    sigma_sc=None):
        """
        Commonly used function to augment hand poses
        :param img: image
        :param gt3Dcrop: 3D annotations
        :param com: center of mass in image coordinates (x,y,z)
        :param cube: cube
        :param aug_modes: augmentation modes
        :param hd: hand detector
        :param normZeroOne: normalization
        :return: image, 3D annotations, com, cube
        """

        assert len(img.shape) == 2
        assert isinstance(aug_modes, list)

        if sigma_com is None:
            sigma_com = 5.

        if sigma_sc is None:
            sigma_sc = 0.02

        if normZeroOne is True:
            img = img * cube[2] + (com[2] - (cube[2] / 2.))
        else:
            img = img * (cube[2] / 2.) + com[2]
        premax = img.max()

        mode = self.rng.randint(0, len(aug_modes))
        off = self.rng.randn(3) * sigma_com  # +-px/mm
        rot = self.rng.uniform(0, 360)
        sc = abs(1. + self.rng.randn() * sigma_sc)
        if aug_modes[mode] == 'com':
            imgD, new_joints3D, com, M = hd.moveCoM(img.astype('float32'), cube, com, off, gt3Dcrop, M, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'rot':
            imgD, new_joints3D, _ = hd.rotateHand(img.astype('float32'), cube, com, rot, gt3Dcrop, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'sc':
            imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'), cube, com, sc, gt3Dcrop, M, pad_value=0)
            curLabel = new_joints3D / (cube[2] / 2.)
        elif aug_modes[mode] == 'none':
            imgD = img
            curLabel = gt3Dcrop / (cube[2] / 2.)
        else:
            raise NotImplementedError()

        if normZeroOne is True:
            imgD[imgD == premax] = com[2] + (cube[2] / 2.)
            imgD[imgD == 0] = com[2] + (cube[2] / 2.)
            imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
            imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
            imgD -= (com[2] - (cube[2] / 2.))
            imgD /= cube[2]
        else:
            imgD[imgD == premax] = com[2] + (cube[2] / 2.)
            imgD[imgD == 0] = com[2] + (cube[2] / 2.)
            imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
            imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
            imgD -= com[2]
            imgD /= (cube[2] / 2.)

        return imgD, curLabel, numpy.asarray(cube), com, M
