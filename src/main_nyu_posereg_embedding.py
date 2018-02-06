"""This is the main file for training hand joint classifier on NYU dataset

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
import os
import cPickle
from sklearn.decomposition import PCA
from trainer.poseregnettrainer import PoseRegNetTrainer, PoseRegNetTrainerParams
from net.poseregnet import PoseRegNetParams, PoseRegNet
from data.importers import NYUImporter
from data.dataset import NYUDataset
from util.handdetector import HandDetector
from util.handpose_evaluation import NYUHandposeEvaluation
from data.transformations import transformPoints2D
from net.hiddenlayer import HiddenLayer, HiddenLayerParams

if __name__ == '__main__':

    eval_prefix = 'NYU_EMB_t0nF8mp421fD553h1024_PCA30_AUGMENT'
    if not os.path.exists('./eval/'+eval_prefix+'/'):
        os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(23455)

    print("create data")
    aug_modes = ['com', 'rot', 'none']  # 'sc',

    comref = None  # "./eval/NYU_COM_AUGMENT/net_NYU_COM_AUGMENT.pkl"
    docom = False
    di = NYUImporter('../data/NYU/', refineNet=comref)
    Seq1 = di.loadSequence('train', shuffle=True, rng=rng, docom=docom)
    trainSeqs = [Seq1]

    Seq2_1 = di.loadSequence('test_1', docom=docom)
    Seq2_2 = di.loadSequence('test_2', docom=docom)
    testSeqs = [Seq2_1, Seq2_2]

    # create training data
    trainDataSet = NYUDataset(trainSeqs)
    train_data, train_gt3D = trainDataSet.imgStackDepthOnly('train')
    train_data_cube = numpy.asarray([Seq1.config['cube']]*train_data.shape[0], dtype='float32')
    train_data_com = numpy.asarray([d.com for d in Seq1.data], dtype='float32')
    train_data_M = numpy.asarray([da.T for da in Seq1.data], dtype='float32')
    train_gt3Dcrop = numpy.asarray([d.gt3Dcrop for d in Seq1.data], dtype='float32')

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    valDataSet = NYUDataset(testSeqs)
    val_data, val_gt3D = valDataSet.imgStackDepthOnly('test_1')

    testDataSet = NYUDataset(testSeqs)
    test_data1, test_gt3D1 = testDataSet.imgStackDepthOnly('test_1')
    test_data2, test_gt3D2 = testDataSet.imgStackDepthOnly('test_2')

    print train_gt3D.max(), test_gt3D1.max(), train_gt3D.min(), test_gt3D1.min()
    print train_data.max(), test_data1.max(), train_data.min(), test_data1.min()

    imgSizeW = train_data.shape[3]
    imgSizeH = train_data.shape[2]
    nChannels = train_data.shape[1]

    ####################################
    # convert data to embedding
    pca = PCA(n_components=30)
    pca.fit(HandDetector.sampleRandomPoses(di, rng, train_gt3Dcrop, train_data_com, train_data_cube, 1e6,
                                           aug_modes).reshape((-1, train_gt3D.shape[1]*3)))
    train_gt3D_embed = pca.transform(train_gt3D.reshape((train_gt3D.shape[0], train_gt3D.shape[1]*3)))
    test_gt3D_embed1 = pca.transform(test_gt3D1.reshape((test_gt3D1.shape[0], test_gt3D1.shape[1]*3)))
    test_gt3D_embed2 = pca.transform(test_gt3D2.reshape((test_gt3D2.shape[0], test_gt3D2.shape[1]*3)))
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
    poseNetTrainer.addStaticData({'val_data_y3D': val_gt3D})
    poseNetTrainer.addStaticData({'pca_data': pca.components_, 'mean_data': pca.mean_})
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
    fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_cost.png')

    fig = plt.figure()
    plt.plot(numpy.asarray(val_errs).T)
    plt.show(block=False)
    fig.savefig('./eval/'+eval_prefix+'/'+eval_prefix+'_errs.png')

    # save results
    poseNet.save("./eval/{}/net_{}.pkl".format(eval_prefix, eval_prefix))
    # poseNet.load("./eval/{}/net_{}.pkl".format(eval_prefix, eval_prefix))

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
    poseNet.save("./eval/{}/network_prior.pkl".format(eval_prefix))

    ###################################################################
    print("Testing ...")
    gt3D = []
    joints = []
    for seq in testSeqs:
        gt3D.extend([j.gt3Dorig for j in seq.data])
        test_data, _ = testDataSet.imgStackDepthOnly(seq.name)
        jts_embed = poseNet.computeOutput(test_data)
        # Backtransform from embedding
        # jts = pca.inverse_transform(jts_embed)
        jts = jts_embed
        for i in xrange(test_data.shape[0]):
            joints.append(jts[i].reshape((-1, 3))*(seq.config['cube'][2]/2.) + seq.data[i].com)

    joints = numpy.array(joints)

    hpe = NYUHandposeEvaluation(gt3D, joints)
    hpe.subfolder += '/'+eval_prefix+'/'
    print("Train samples: {}, test samples: {}".format(train_data.shape[0], len(gt3D)))
    print("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError()))
    print("{}".format([hpe.getJointMeanError(j) for j in range(joints[0].shape[0])]))
    print("{}".format([hpe.getJointMaxError(j) for j in range(joints[0].shape[0])]))

    # save results
    cPickle.dump(joints, open("./eval/{}/result_{}_{}.pkl".format(eval_prefix, os.path.split(__file__)[1], eval_prefix), "wb"), protocol=cPickle.HIGHEST_PROTOCOL)

    print "Testing baseline"

    #################################
    # BASELINE
    # Load the evaluation
    data_baseline = di.loadBaseline('../data/NYU/test/test_predictions.mat', numpy.asarray(gt3D))

    hpe_base = NYUHandposeEvaluation(gt3D, data_baseline)
    hpe_base.subfolder += '/'+eval_prefix+'/'
    print("Mean error: {}mm".format(hpe_base.getMeanError()))
    hpe.plotEvaluation(eval_prefix, methodName='Our regr', baseline=[('Tompson et al.', hpe_base)])

    ind = 0
    for i in testSeqs[0].data:
        if ind % 20 != 0:
            ind += 1
            continue
        jtI = transformPoints2D(di.joints3DToImg(joints[ind]), i.T)
        hpe.plotResult(i.dpt, i.gtcrop, jtI, "{}_{}".format(eval_prefix, ind))
        ind += 1
