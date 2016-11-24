"""This is the main file for training hand detection refinement on ICVL dataset

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
import matplotlib
matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
from net.scalenet import ScaleNetParams, ScaleNet
from trainer.scalenettrainer import ScaleNetTrainerParams, ScaleNetTrainer
from util.handdetector import HandDetector

import theano
import os
import cPickle
import sys
from data.importers import ICVLImporter
from data.dataset import ICVLDataset
from util.handpose_evaluation import ICVLHandposeEvaluation
import cv2

if __name__ == '__main__':

    eval_prefix = 'ICVL_COM'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    floatX = theano.config.floatX  # @UndefinedVariable

    rng = numpy.random.RandomState(23455)

    print("create data")

    di = ICVLImporter('../data/ICVL/')
    Seq1 = di.loadSequence('train', ['0'],shuffle=True,rng=rng,docom=True)
    trainSeqs = [Seq1]

    Seq2 = di.loadSequence('test_seq_1',docom=True)
    testSeqs = [Seq2]

    # create training data
    trainDataSet = ICVLDataset(trainSeqs)
    train_data, train_gt3D = trainDataSet.imgStackDepthOnly('train')

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    testDataSet = ICVLDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_seq_1')

    val_data = test_data
    val_gt3D = test_gt3D

    ####################################
    # resize data
    dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    train_data2 = train_data[:, :, ystart:yend, xstart:xend]

    dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    train_data4 = train_data[:, :, ystart:yend, xstart:xend]

    dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    val_data2 = val_data[:, :, ystart:yend, xstart:xend]

    dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    val_data4 = val_data[:, :, ystart:yend, xstart:xend]

    dsize = (int(train_data.shape[2]//2), int(train_data.shape[3]//2))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    test_data2 = test_data[:, :, ystart:yend, xstart:xend]

    dsize = (int(train_data.shape[2]//4), int(train_data.shape[3]//4))
    xstart = int(train_data.shape[2]/2-dsize[0]/2)
    xend = xstart + dsize[0]
    ystart = int(train_data.shape[3]/2-dsize[1]/2)
    yend = ystart + dsize[1]
    test_data4 = test_data[:, :, ystart:yend, xstart:xend]

    print train_gt3D.max(), test_gt3D.max(), train_gt3D.min(), test_gt3D.min()
    print train_data.max(), test_data.max(), train_data.min(), test_data.min()

    imgSizeW = train_data.shape[3]
    imgSizeH = train_data.shape[2]
    nChannels = train_data.shape[1]

    #############################################################################
    print("create network")
    batchSize = 64
    poseNetParams = ScaleNetParams(type=1, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                   resizeFactor=2, numJoints=1, nDims=3)
    poseNet = ScaleNet(rng, cfgParams=poseNetParams)

    poseNetTrainerParams = ScaleNetTrainerParams()
    poseNetTrainerParams.batch_size = batchSize
    poseNetTrainerParams.learning_rate = 0.0005
    poseNetTrainerParams.weightreg_factor = 0.0001
    poseNetTrainerParams.lr_of_ep = lambda ep: poseNetTrainerParams.learning_rate/(1+0.1*ep)

    print("setup trainer")
    poseNetTrainer = ScaleNetTrainer(poseNet, poseNetTrainerParams, rng)
    poseNetTrainer.setData(train_data, train_gt3D[:, 0, :], val_data, val_gt3D[:, 0, :])
    poseNetTrainer.addStaticData({'val_data_x1': val_data2, 'val_data_x2': val_data4})
    poseNetTrainer.addManagedData({'train_data_x1': train_data2, 'train_data_x2': train_data4})
    poseNetTrainer.compileFunctions()

    ###################################################################
    # TRAIN
    train_res = poseNetTrainer.train(n_epochs=100, storeFilters=True)
    train_costs = train_res[0]
    wvals = train_res[1]
    val_errs = train_res[2]

    # plot cost
    fig = plt.figure()
    plt.semilogy(train_costs)
    plt.show(block=False)
    fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_cost.png')

    fig = plt.figure()
    plt.semilogy(val_errs)
    plt.show(block=False)
    fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_errs.png')

    # save results
    poseNet.save("./eval/{}/net_{}.pkl".format(eval_prefix, eval_prefix))
    # poseNet.load("./eval/{}/net_{}.pkl".format(eval_prefix,eval_prefix))

    ####################################################
    # TEST
    print("Testing ...")
    gt3D = [j.gt3Dorig[0].reshape(1, 3) for j in testSeqs[0].data]
    jts = poseNet.computeOutput([test_data, test_data2, test_data4])
    joints = []
    for i in xrange(test_data.shape[0]):
        joints.append(jts[i].reshape(1, 3)*(testSeqs[0].config['cube'][2]/2.) + testSeqs[0].data[i].com)

    hpe = ICVLHandposeEvaluation(gt3D, joints)
    hpe.subfolder += '/'+eval_prefix+'/'
    mean_error = hpe.getMeanError()
    max_error = hpe.getMaxError()
    print("Mean error: {}mm, max error: {}mm".format(mean_error, max_error))

    # save results
    cPickle.dump(joints, open("./eval/{}/result_{}_{}.pkl".format(eval_prefix,os.path.split(__file__)[1],eval_prefix), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)

    print "Testing baseline"

    #################################
    # BASELINE
    # Load the evaluation
    data_baseline = di.loadBaseline('../data/ICVL/LRF_Results_seq_1.txt')

    hpe_base = ICVLHandposeEvaluation(gt3D, numpy.asarray(data_baseline)[:,0,:].reshape((len(gt3D),1,3)))
    hpe_base.subfolder += '/'+eval_prefix+'/'
    print("Mean error: {}mm".format(hpe_base.getMeanError()))

    com = [j.com for j in testSeqs[0].data]
    hpe_com = ICVLHandposeEvaluation(gt3D, numpy.asarray(com).reshape((len(gt3D),1,3)))
    hpe_com.subfolder += '/'+eval_prefix+'/'
    print("Mean error: {}mm".format(hpe_com.getMeanError()))

