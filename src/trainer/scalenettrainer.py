"""Network trainer for multiscale regression networks.

ScaleNetTrainer provides interface for training regressors for
estimating the hand pose.
ScaleNetTrainerParams is the parametrization of the ScaleNetTrainer.

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

from trainer.nettrainer import NetTrainerParams, NetTrainer
from trainer.optimizer import Optimizer


__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class ScaleNetTrainerParams(NetTrainerParams):
    def __init__(self):
        super(ScaleNetTrainerParams, self).__init__()


class ScaleNetTrainer(NetTrainer):
    """
    classdocs
    """

    def __init__(self, poseNet=None, cfgParams=None, rng=None, subfolder='./eval/', numChunks=1):
        """
        Constructor
        
        :param poseNet: initialized DescriptorNet
        :param cfgParams: initialized PoseRegNetTrainerParams
        """

        super(ScaleNetTrainer, self).__init__(cfgParams, 8, subfolder, numChunks)
        self.poseNet = poseNet
        self.rng = rng

        if not isinstance(cfgParams, ScaleNetTrainerParams):
            raise ValueError("cfgParams must be an instance of ScaleNetTrainerParams")

        self.setupFunctions()

    def setupFunctions(self):
        floatX = theano.config.floatX  # @UndefinedVariable

        dnParams = self.poseNet.cfgParams

        # params
        self.learning_rate = T.scalar('learning_rate')
        self.momentum = T.scalar('momentum')

        # input
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = []
        for i in range(self.poseNet.cfgParams.numInputs):
            self.x.append(self.poseNet.inputVar[i])

        # targets
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            y = T.vector('y')  # R^D
        elif self.poseNet.cfgParams.numJoints == 1:
            y = T.matrix('y')  # R^Dx3
        else:
            y = T.tensor3('y')  # R^Dx16x3

        # L2 error
        if self.poseNet.cfgParams.numJoints == 1 and self.poseNet.cfgParams.nDims == 1:
            cost = T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims)) - y)
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
            errors = T.sqrt(T.sqr(T.reshape(self.poseNet.output,(self.cfgParams.batch_size,self.poseNet.cfgParams.nDims))-y))
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
        givens_train = {self.x[0]: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        for i in range(1, self.poseNet.cfgParams.numInputs):
            givens_train[self.x[i]] = getattr(self, 'train_data_x'+str(i))[self.index * batch_size:(self.index + 1) * batch_size]
        givens_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        print("compiling train_model() ... ")
        self.train_model = theano.function(inputs=[self.index, self.learning_rate],
                                           outputs=self.cost,
                                           updates=self.updates,
                                           givens=givens_train)

        print("done.")

        print("compiling test_model_on_train() ... ")
        batch_size = self.cfgParams.batch_size
        givens_test_on_train = {self.x[0]: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        for i in range(1, self.poseNet.cfgParams.numInputs):
            givens_test_on_train[self.x[i]] = getattr(self, 'train_data_x'+str(i))[self.index * batch_size:(self.index + 1) * batch_size]
        givens_test_on_train[self.y] = self.train_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        self.test_model_on_train = theano.function(inputs=[self.index],
                                                   outputs=self.errors,
                                                   givens=givens_test_on_train)
        print("done.")

    def setupValidate(self):

        batch_size = self.cfgParams.batch_size
        givens_val = {self.x[0]: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        for i in range(1, self.poseNet.cfgParams.numInputs):
            givens_val[self.x[i]] = getattr(self, 'val_data_x'+str(i))[self.index * batch_size:(self.index + 1) * batch_size]
        givens_val[self.y] = self.val_data_y[self.index * batch_size:(self.index + 1) * batch_size]

        givens_val_cost = {self.x[0]: self.val_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        for i in range(1, self.poseNet.cfgParams.numInputs):
            givens_val_cost[self.x[i]] = getattr(self, 'val_data_x'+str(i))[self.index * batch_size:(self.index + 1) * batch_size]
        givens_val_cost[self.y] = self.val_data_y[self.index * batch_size:(self.index + 1) * batch_size]
        print("compiling validation_cost() ... ")
        self.validation_cost = theano.function(inputs=[self.index],
                                               outputs=self.cost,
                                               givens=givens_val_cost)
        print("done.")
        self.validation_observer.append(self.validation_cost)

        print("compiling validation_error() ... ")
        self.validation_error = theano.function(inputs=[self.index],
                                                outputs=self.errors,
                                                givens=givens_val)
        print("done.")
        self.validation_observer.append(self.validation_error)

    def setupDebugFunctions(self):
        batch_size = self.cfgParams.batch_size

        print("compiling compute_train_descr() ... ")
        givens_train_descr = {self.x[0]: self.train_data_x[self.index * batch_size:(self.index + 1) * batch_size]}
        for i in range(1, self.poseNet.cfgParams.numInputs):
            givens_train_descr[self.x[i]] = getattr(self, 'train_data_x'+str(i))[self.index * batch_size:(self.index + 1) * batch_size]
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
                com = self.train_data_comDBlast[i].copy()
                cube = self.train_data_cubeDBlast[i].copy()
                gt3D = self.train_data_yDBlast[i].copy().reshape((1, 3))
            else:
                img = self.train_data_xDB[i, 0].copy()
                com = self.train_data_comDB[i].copy()
                cube = self.train_data_cubeDB[i].copy()
                gt3D = self.train_data_yDB[i].copy().reshape((1, 3))

            imgD, gt3D, cube, com2D, M = self.augmentCrop(
                img, gt3D*(cube[2] / 2.), macro_params['args']['di'].joint3DToImg(com), cube, numpy.eye(3),
                macro_params['args']['aug_modes'], macro_params['args']['hd'], macro_params['args']['normZeroOne'],
                sigma_com=(macro_params['args']['sigma_com'] if 'sigma_com' in macro_params['args'] else None),
                sigma_sc=(macro_params['args']['sigma_sc'] if 'sigma_sc' in macro_params['args'] else None))
            com = macro_params['args']['di'].jointImgTo3D(com2D)

            new_data['train_data_x'][idx] = imgD
            new_data['train_data_y'][idx] = gt3D.flatten()

        dsize = (int(new_data['train_data_x'].shape[2]//2), int(new_data['train_data_x'].shape[3]//2))
        xstart = int(new_data['train_data_x'].shape[2]/2-dsize[0]/2)
        xend = xstart + dsize[0]
        ystart = int(new_data['train_data_x'].shape[3]/2-dsize[1]/2)
        yend = ystart + dsize[1]
        new_data['train_data_x1'][:] = new_data['train_data_x'][:, :, ystart:yend, xstart:xend]

        dsize = (int(new_data['train_data_x'].shape[2]//4), int(new_data['train_data_x'].shape[3]//4))
        xstart = int(new_data['train_data_x'].shape[2]/2-dsize[0]/2)
        xend = xstart + dsize[0]
        ystart = int(new_data['train_data_x'].shape[3]/2-dsize[1]/2)
        yend = ystart + dsize[1]
        new_data['train_data_x2'][:] = new_data['train_data_x'][:, :, ystart:yend, xstart:xend]
