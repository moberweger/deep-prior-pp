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

import numpy
import theano
import theano.tensor as T
from net.poollayer import PoolLayer
from net.poseregnet import PoseRegNet, PoseRegNetParams
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

    def __init__(self, poseNet=None, cfgParams=None, rng=None, subfolder='./eval/', numChunks=1):
        """
        Constructor
        
        :param poseNet: initialized DescriptorNet
        :param cfgParams: initialized PoseRegNetTrainerParams
        """
        super(PoseRegNetTrainer, self).__init__(cfgParams, 5, subfolder, numChunks)
        self.poseNet = poseNet
        self.rng = rng

        if not isinstance(cfgParams, PoseRegNetTrainerParams):
            raise ValueError("cfgParams must be an instance of PoseRegNetTrainerParams")

        self.setupFunctions()

    def setupFunctions(self):
        floatX = theano.config.floatX  # @UndefinedVariable

        dnParams = self.poseNet.cfgParams

        # params
        self.learning_rate = T.scalar('learning_rate')
        self.momentum = T.scalar('momentum')

        # input
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = self.poseNet.inputVar

        # targets
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            y = T.vector('y')  # R^D
        elif self.poseNet.cfgParams.numJoints == 1:
            y = T.matrix('y')  # R^Dx3
        else:
            y = T.tensor3('y')  # R^Dx16x3

        # L2 error
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y).mean(axis=1)
        elif self.poseNet.cfgParams.numJoints == 1:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y).sum(axis=1)
        else:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.numJoints,self.poseNet.cfgParams.nDims))-y).sum(axis=2).mean(axis=1) # error is sum of all joints

        self.cost = cost.mean()  # The cost to minimize

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

        # evaluation errors
        self.y_eval = T.tensor3('y')  # R^Dx16x3
        self.pca = T.matrix('pca')
        self.mean = T.vector('mean')
        self.errors_avg = T.sqrt(T.sqr(T.reshape(T.dot(self.poseNet.output, self.pca)+self.mean,(self.cfgParams.batch_size, self.pca.shape[1]//3, 3))-self.y_eval).sum(axis=2)).mean(axis=1).mean()
        self.errors_max = T.sqrt(T.sqr(T.reshape(T.dot(self.poseNet.output, self.pca)+self.mean,(self.cfgParams.batch_size, self.pca.shape[1]//3, 3))-self.y_eval).sum(axis=2)).max(axis=1).max()

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
        self.updates = opt.ADAM(self.learning_rate)

        batch_size = self.cfgParams.batch_size
        givens_train = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling train_model() ... ")
        self.train_model = theano.function(inputs=[self.index, self.learning_rate],
                                           outputs=self.cost,
                                           updates=self.updates,
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

        print("compiling validation_cost() ... ")
        self.validation_cost = theano.function(inputs=[self.index],
                                               outputs=self.cost,
                                               givens=givens_val)
        print("done.")
        self.validation_observer.append(self.validation_cost)

        givens_val_err = {self.x: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        givens_val_err[self.y] = self.val_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling validation_error() ... ")
        self.validation_error = theano.function(inputs=[self.index],
                                                outputs=self.errors,
                                                givens=givens_val_err)
        print("done.")
        self.validation_observer.append(self.validation_error)

        print("compiling validation_error_avg() ... ")
        if hasattr(self, 'val_data_y3D'):
            givens_val2 = {self.x: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
            givens_val2[self.y_eval] = self.val_data_y3D[self.index * batch_size:(self.index + 1) * batch_size]
            givens_val2[self.pca] = self.pca_data
            givens_val2[self.mean] = self.mean_data
            self.validation_error_avg = theano.function(inputs=[self.index],
                                                    outputs=self.errors_avg,
                                                    givens=givens_val2)
            self.validation_error_max = theano.function(inputs=[self.index],
                                                    outputs=self.errors_max,
                                                    givens=givens_val2)
            self.validation_observer.append(self.validation_error_avg)
            self.validation_observer.append(self.validation_error_max)
        print("done.")

    def setupDebugFunctions(self):
        batch_size = self.cfgParams.batch_size

        print("compiling compute_train_descr() ... ")
        givens_train_descr = {self.x: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        self.compute_train_descr = theano.function(inputs=[self.index],
                                                   outputs=self.poseNet.output,
                                                   givens=givens_train_descr)
        print("done.")

    def augment_poses(self, macro_params, macro_idx, last, tidxs, idxs, new_data):
        # augment the training data within current data range
        for idx, i in zip(tidxs, idxs):
            # com now in image coordinates
            if (self.getNumMacroBatches() > 1) and (last is True):
                img = self.train_data_xDBlast[i, 0].copy()
                com = macro_params['args']['di'].joint3DToImg(self.train_data_comDBlast[i])
                cube = self.train_data_cubeDBlast[i].copy()
                if 'proj' in macro_params['args'] and macro_params['args']['proj'] is not None:
                    gt3Dcrop = self.train_gt3DcropDBlast[i].copy()
                else:
                    gt3Dcrop = self.train_data_yDBlast[i].copy().reshape((-1, 3)) * (cube[2] / 2.)
            else:
                img = self.train_data_xDB[i, 0].copy()
                com = macro_params['args']['di'].joint3DToImg(self.train_data_comDB[i])
                cube = self.train_data_cubeDB[i].copy()
                if 'proj' in macro_params['args'] and macro_params['args']['proj'] is not None:
                    gt3Dcrop = self.train_gt3DcropDB[i].copy()
                else:
                    gt3Dcrop = self.train_data_yDB[i].copy().reshape((-1, 3)) * (cube[2] / 2.)

            imgD, curLabel, _, _, _ = self.augmentCrop(
                img, gt3Dcrop, com, cube, numpy.eye(3), macro_params['args']['aug_modes'],
                macro_params['args']['hd'], macro_params['args']['normZeroOne'])

            # import scipy
            # scipy.misc.imshow(numpy.concatenate([train_data_xDB[i+start_idx, 0], imgD], axis=0))

            if 'binarizeImage' in macro_params['args']:
                if macro_params['args']['binarizeImage'] is True:
                    imgD[imgD < 0.5] = 0
                    imgD[imgD >= 0.5] = 1
            new_data['train_data_x'][idx] = imgD
            if 'proj' in macro_params['args'] and macro_params['args']['proj'] is not None:
                # check for projection deep prior
                new_data['train_data_y'][idx] = macro_params['args']['proj'].transform(curLabel.reshape(1, -1))[0]
            else:
                new_data['train_data_y'][idx] = curLabel.reshape(new_data['train_data_y'][idx].shape)
