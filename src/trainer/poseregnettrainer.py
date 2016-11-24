"""Network trainer for regression networks.

PoseRegNetTrainer provides interface for training regressors for
estimating the hand pose.
PoseRegNetTrainerParams is the parametrization of the PoseRegNetTrainer.

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

import theano
import theano.tensor as T
from net.poollayer import PoolLayer
from net.poseregnet import PoseRegNet
from trainer.nettrainer import NetTrainerParams, NetTrainer
from trainer.optimizer import Optimizer

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class PoseRegNetTrainerParams(NetTrainerParams):
    def __init__(self):
        super(PoseRegNetTrainerParams, self).__init__()


class PoseRegNetTrainer(NetTrainer):
    """
    classdocs
    """

    def __init__(self, poseNet=None, cfgParams=None, rng=None):
        """
        Constructor
        
        :param poseNet: initialized DescriptorNet
        :param cfgParams: initialized PoseRegNetTrainerParams
        """
        super(PoseRegNetTrainer, self).__init__(cfgParams, 3)
        self.poseNet = poseNet
        self.cfgParams = cfgParams
        self.rng = rng

        if not isinstance(cfgParams, PoseRegNetTrainerParams):
            raise ValueError("cfgParams must be an instance of PoseRegNetTrainerParams")

        self.setupFunctions()

    def setupFunctions(self):
        floatX = theano.config.floatX  # @UndefinedVariable

        dnParams = self.poseNet.cfgParams

        # params
        self.learning_rate = T.scalar('learning_rate', dtype=floatX)
        self.momentum = T.scalar('momentum', dtype=floatX)

        # input
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = self.poseNet.inputVar

        # targets
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            y = T.vector('y',dtype=floatX) # R^D
        elif self.poseNet.cfgParams.numJoints == 1:
            y = T.matrix('y',dtype=floatX) # R^Dx3
        else:
            y = T.tensor3('y',dtype=floatX) # R^Dx16x3

        # L2 error
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y).mean(axis=1)
        elif self.poseNet.cfgParams.numJoints == 1:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y).sum(axis=1)
        else:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.numJoints,self.poseNet.cfgParams.nDims))-y).sum(axis=2).mean(axis=1) # error is sum of all joints

        self.cost = cost.mean() # The cost to minimize

        # weight vector length for regularization (weight decay)       
        totalWeightVectorLength = 0
        for W in self.poseNet.weights:
            totalWeightVectorLength += self.cfgParams.weightreg_factor * (W ** 2).sum()

        if not self.poseNet.hasDropout():
            self.cost += totalWeightVectorLength  # + weight vector norm

        # create a list of gradients for all model parameters
        self.params = self.poseNet.params
        self.grads = T.grad(self.cost, self.params)

        # euclidean mean errors over all joints
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            errors = T.sqrt(T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y)).mean(axis=1)
        elif self.poseNet.cfgParams.numJoints == 1:
            errors = T.sqrt(T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y).sum(axis=1))
        else:
            errors = T.sqrt(T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.numJoints,self.poseNet.cfgParams.nDims))-y).sum(axis=2)).mean(axis=1)

        # mean error over full set
        self.errors = errors.mean()

        # store stuff                    
        self.y = y

    def compileFunctions(self, compileDebugFcts=False):
        # TRAIN
        self.setupTrain()

        # DEBUG
        self.compileDebugFcts = compileDebugFcts
        if compileDebugFcts:
            self.setupDebugFunctions()

        # VALIDATE
        self.setupValidate()

    def setupTrain(self):
        # train_model is a function that updates the model parameters by SGD
        opt = Optimizer(self.grads, self.params)
        updates = opt.RMSProp(self.learning_rate, 0.9, 1.0/100.)

        batch_size = self.cfgParams.batch_size
        givens_train = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling train_model() ... ")
        self.train_model = theano.function(inputs=[self.index, self.learning_rate],
                                           outputs=self.cost,
                                           updates=updates,
                                           givens=givens_train)
        print("done.")

        print("compiling test_model_on_train() ... ")
        batch_size = self.cfgParams.batch_size
        givens_test_on_train = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_test_on_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        self.test_model_on_train = theano.function(inputs=[self.index],
                                                   outputs=self.errors,
                                                   givens=givens_test_on_train)
        print("done.")

    def setupValidate(self):

        batch_size = self.cfgParams.batch_size
        givens_val = {self.x: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_val[self.y] = self.val_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling validation_error() ... ")
        self.validation_error = theano.function(inputs=[self.index],
                                                outputs=self.errors,
                                                givens=givens_val)
        print("done.")

        print("compiling validation_cost() ... ")
        self.validation_cost = theano.function(inputs=[self.index],
                                               outputs=self.cost,
                                               givens=givens_val)
        print("done.")

        # debug and so
        print("compiling compute_val_descr() ... ")
        givens_val_descr = {self.x: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        self.compute_val_descr = theano.function(inputs=[self.index],
                                                 outputs=self.poseNet.output,
                                                 givens=givens_val_descr)
        print("done.")

    def setupDebugFunctions(self):
        batch_size = self.cfgParams.batch_size

        print("compiling compute_train_descr() ... ")
        givens_train_descr = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        self.compute_train_descr = theano.function(inputs=[self.index],
                                                   outputs=self.poseNet.output,
                                                   givens=givens_train_descr)
        print("done.")
