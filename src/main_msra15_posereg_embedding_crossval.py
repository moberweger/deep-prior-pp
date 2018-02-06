"""This is the main file for training hand joint classifier on MSRA dataset

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
import gc
from data.transformations import transformPoints2D
from net.hiddenlayer import HiddenLayer, HiddenLayerParams
from util.handdetector import HandDetector
from util.helpers import shuffle_many_inplace

matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
import os
import cPickle
from trainer.poseregnettrainer import PoseRegNetTrainer, PoseRegNetTrainerParams
from net.poseregnet import PoseRegNetParams, PoseRegNet
from data.importers import MSRA15Importer
from data.dataset import MSRA15Dataset
from util.handpose_evaluation import MSRAHandposeEvaluation
from sklearn.decomposition import PCA

if __name__ == '__main__':

    eval_prefix = 'MSRA15_EMB_t0nF8mp421fD553h1024_PCA30_AUGMENT_CV'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/MSRA15_COM_AUGMENT/net_MSRA15_COM_AUGMENT.pkl"
    docom = False
    di = MSRA15Importer('../data/MSRA15/', refineNet=comref)
    Seq0 = di.loadSequence('P0', shuffle=True, rng=rng, docom=docom)
    Seq1 = di.loadSequence('P1', shuffle=True, rng=rng, docom=docom)
    Seq2 = di.loadSequence('P2', shuffle=True, rng=rng, docom=docom)
    Seq3 = di.loadSequence('P3', shuffle=True, rng=rng, docom=docom)
    Seq4 = di.loadSequence('P4', shuffle=True, rng=rng, docom=docom)
    Seq5 = di.loadSequence('P5', shuffle=True, rng=rng, docom=docom)
    Seq6 = di.loadSequence('P6', shuffle=True, rng=rng, docom=docom)
    Seq7 = di.loadSequence('P7', shuffle=True, rng=rng, docom=docom)
    Seq8 = di.loadSequence('P8', shuffle=True, rng=rng, docom=docom)
    seqs = [Seq0, Seq1, Seq2, Seq3, Seq4, Seq5, Seq6, Seq7, Seq8]

    for icv in xrange(len(seqs)):
        trainSeqs = [s for i, s in enumerate(seqs) if i != icv]
        testSeqs = [seqs[icv]]
        print "training: {}".format(' '.join([s.name for s in trainSeqs]))
        print "testing: {}".format(' '.join([s.name for s in testSeqs]))

        # create training data
        trainDataSet = MSRA15Dataset(trainSeqs, localCache=False)
        nSamp = numpy.sum([len(s.data) for s in trainSeqs])
        d1, g1 = trainDataSet.imgStackDepthOnly(trainSeqs[0].name)
        train_data = numpy.ones((nSamp, d1.shape[1], d1.shape[2], d1.shape[3]), dtype='float32')
        train_gt3D = numpy.ones((nSamp, g1.shape[1], g1.shape[2]), dtype='float32')
        train_data_cube = numpy.ones((nSamp, 3), dtype='float32')
        train_data_com = numpy.ones((nSamp, 3), dtype='float32')
        train_data_M = numpy.ones((nSamp, 3, 3), dtype='float32')
        train_gt3Dcrop = numpy.ones_like(train_gt3D)
        del d1, g1
        gc.collect()
        gc.collect()
        gc.collect()
        oldIdx = 0
        for seq in trainSeqs:
            d, g = trainDataSet.imgStackDepthOnly(seq.name)
            train_data[oldIdx:oldIdx+d.shape[0]] = d
            train_gt3D[oldIdx:oldIdx+d.shape[0]] = g
            train_data_cube[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([seq.config['cube']]*d.shape[0])
            train_data_com[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([da.com for da in seq.data])
            train_data_M[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([da.T for da in seq.data])
            train_gt3Dcrop[oldIdx:oldIdx+d.shape[0]] = numpy.asarray([da.gt3Dcrop for da in seq.data])
            oldIdx += d.shape[0]
            del d, g
            gc.collect()
            gc.collect()
            gc.collect()
        shuffle_many_inplace([train_data, train_gt3D, train_data_cube, train_data_com, train_gt3Dcrop, train_data_M], random_state=rng)

        mb = (train_data.nbytes) / (1024 * 1024)
        print("data size: {}Mb".format(mb))

        testDataSet = MSRA15Dataset(testSeqs)
        test_data, test_gt3D = testDataSet.imgStackDepthOnly(testSeqs[0].name)

        val_data = test_data
        val_gt3D = test_gt3D

        print train_gt3D.max(), test_gt3D.max(), train_gt3D.min(), test_gt3D.min()
        print train_data.max(), test_data.max(), train_data.min(), test_data.min()

        imgSizeW = train_data.shape[3]
        imgSizeH = train_data.shape[2]
        nChannels = train_data.shape[1]

        ####################################
        # convert data to embedding
        # diaboloNet = cPickle.load(open("./eval/ICVL_DIA_t0D8h200-8-200/HandCascade_base_cache_ICVL_DIA_t0D8h200-8-200.pkl", "rb"))
        # train_gt3D_embed = diaboloNet.computeEmbedding(train_gt3D.reshape((train_gt3D.shape[0],train_gt3D.shape[1]*3)))
        # test_gt3D_embed = diaboloNet.computeEmbedding(test_gt3D.reshape((test_gt3D.shape[0],test_gt3D.shape[1]*3)))
        # val_gt3D_embed = diaboloNet.computeEmbedding(val_gt3D.reshape((val_gt3D.shape[0],val_gt3D.shape[1]*3)))

        pca = PCA(n_components=30)
        pca.fit(HandDetector.sampleRandomPoses(di, rng, train_gt3Dcrop, train_data_com, train_data_cube, 1e6,
                                               aug_modes).reshape((-1, train_gt3D.shape[1]*3)))
        train_gt3D_embed = pca.transform(train_gt3D.reshape((train_gt3D.shape[0], train_gt3D.shape[1]*3)))
        test_gt3D_embed = pca.transform(test_gt3D.reshape((test_gt3D.shape[0], test_gt3D.shape[1]*3)))
        val_gt3D_embed = pca.transform(val_gt3D.reshape((val_gt3D.shape[0], val_gt3D.shape[1]*3)))

        ############################################################################
        print("create network")
        batchSize = 128
        poseNetParams = PoseRegNetParams(type=0, nChan=nChannels, wIn=imgSizeW, hIn=imgSizeH, batchSize=batchSize,
                                         numJoints=1, nDims=train_gt3D_embed.shape[1])
        poseNet = PoseRegNet(rng, cfgParams=poseNetParams)

        poseNetTrainerParams = PoseRegNetTrainerParams()
        poseNetTrainerParams.batch_size = batchSize
        poseNetTrainerParams.learning_rate = 0.001
        poseNetTrainerParams.weightreg_factor = 0.0
        poseNetTrainerParams.force_macrobatch_reload = True
        poseNetTrainerParams.para_augment = True
        poseNetTrainerParams.augment_fun_params = {'fun': 'augment_poses', 'args': {'normZeroOne': False,
                                                                                    'di': di,
                                                                                    'aug_modes': aug_modes,
                                                                                    'hd': HandDetector(train_data[0, 0].copy(), abs(di.fx), abs(di.fy), importer=di),
                                                                                    'proj': pca}}

        print("setup trainer")
        poseNetTrainer = PoseRegNetTrainer(poseNet, poseNetTrainerParams, rng, './eval/'+eval_prefix)
        poseNetTrainer.setData(train_data, train_gt3D_embed, val_data, val_gt3D_embed)
        poseNetTrainer.addManagedData({'train_data_cube': train_data_cube,
                                       'train_data_com': train_data_com,
                                       'train_data_M': train_data_M,
                                       'train_gt3Dcrop': train_gt3Dcrop})
        poseNetTrainer.compileFunctions(compileDebugFcts=False)

        ###################################################################
        # TRAIN
        train_res = poseNetTrainer.train(n_epochs=100)
        train_costs = train_res[0]
        val_errs = train_res[2]

        ###################################################################
        # TEST
        # plot cost
        fig = plt.figure()
        plt.semilogy(train_costs)
        plt.show(block=False)
        fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_cost_{}.png'.format(icv))

        fig = plt.figure()
        plt.semilogy(val_errs)
        plt.show(block=False)
        fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_errs_{}.png'.format(icv))

        # save results
        poseNet.save("./eval/{}/net_{}_{}.pkl".format(eval_prefix, eval_prefix, icv))
        # poseNet.load("./eval/{}/net_{}_{}.pkl".format(eval_prefix, eval_prefix, icv))

        # add prior to network
        cfg = HiddenLayerParams(inputDim=(batchSize, train_gt3D_embed.shape[1]),
                                outputDim=(batchSize, numpy.prod(train_gt3D.shape[1:])), activation=None)
        pcalayer = HiddenLayer(rng, poseNet.layers[-1].output, cfg, layerNum=len(poseNet.layers))
        pcalayer.W.set_value(pca.components_)
        pcalayer.b.set_value(pca.mean_)
        poseNet.layers.append(pcalayer)
        poseNet.output = pcalayer.output
        poseNet.cfgParams.numJoints = train_gt3D.shape[1]
        poseNet.cfgParams.nDims = train_gt3D.shape[2]
        poseNet.cfgParams.outputDim = pcalayer.cfgParams.outputDim
        poseNet.save("./eval/{}/network_prior_{}.pkl".format(eval_prefix, icv))

        ###################################################################

        print("Testing ...")
        gt3D = [j.gt3Dorig for j in testSeqs[0].data]
        joints = []
        jts_embed = poseNet.computeOutput(test_data)
        # Backtransform from embedding
        # jts = diaboloNet.computeOutputFromEmbedding(jts_embed) # calculate pose from codes
        # jts = pca.inverse_transform(jts_embed)
        jts = jts_embed
        for i in range(test_data.shape[0]):
            joints.append(jts[i].reshape((-1, 3))*(testSeqs[0].config['cube'][2]/2.) + testSeqs[0].data[i].com)

        joints = numpy.array(joints)

        hpe = MSRAHandposeEvaluation(gt3D, joints)
        hpe.subfolder += '/'+eval_prefix+'/'
        print("Train samples: {}, test samples: {}".format(train_data.shape[0], len(gt3D)))
        print("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError()))
        print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
        print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

        # save results
        cPickle.dump(gt3D, open("./eval/{}/gt_{}.pkl".format(eval_prefix, icv), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(joints, open("./eval/{}/result_{}.pkl".format(eval_prefix, icv), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)

        print "Testing baseline"

        #################################
        # BASELINE
        hpe.plotEvaluation(eval_prefix+"_{}".format(icv), methodName='Our regr')

        ind = 0
        for i in testSeqs[0].data:
            if ind % 20 != 0:
                ind += 1
                continue
            jtI = transformPoints2D(di.joints3DToImg(joints[ind]), i.T)
            hpe.plotResult(i.dpt, i.gtcrop, jtI, "{}_{}".format(icv, ind))
            ind += 1

        del poseNetTrainer, poseNet
        gc.collect()
        gc.collect()
        gc.collect()

    ###########################################
    # evaluation
    print "Result of cross-validation:"
    all_gt = []
    all_results = []
    for icv in xrange(len(seqs)):
        all_gt.extend(cPickle.load(open("./eval/{}/gt_{}.pkl".format(eval_prefix, icv), "rb")))
        all_results.extend(cPickle.load(open("./eval/{}/result_{}.pkl".format(eval_prefix, icv), "rb")))

    hpe = MSRAHandposeEvaluation(all_gt, all_results)
    hpe.subfolder += '/'+eval_prefix+'/'
    print("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError()))
    print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    hpe.plotEvaluation(eval_prefix, methodName='Our regr')
