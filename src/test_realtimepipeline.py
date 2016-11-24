"""
This is the main file for testing realtime performance.

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

import glob
import numpy
from data.dataset import NYUDataset, ICVLDataset
from net.poseregnet import PoseRegNetParams, PoseRegNet
from net.scalenet import ScaleNetParams, ScaleNet
from util.realtimehandposepipeline import RealtimeHandposePipeline
from data.importers import ICVLImporter, NYUImporter, DepthImporter


__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"

if __name__ == '__main__':

    # di = ICVLImporter('../data/ICVL/')
    # Seq2 = di.loadSequence('test_seq_1')
    # testSeqs = [Seq2]
    #
    # testDataSet = ICVLDataset(testSeqs)
    # test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_seq_1')

    di = NYUImporter('../data/NYU/')
    Seq2 = di.loadSequence('test_1')
    testSeqs = [Seq2]

    testDataSet = NYUDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_1')

    # load trained network
    # poseNetParams = PoseRegNetParams(type=11, nChan=1, wIn=128, hIn=128, batchSize=1, numJoints=16, nDims=3)
    # poseNet = PoseRegNet(numpy.random.RandomState(23455), cfgParams=poseNetParams)
    # poseNet.load("./ICVL_network_prior.pkl")
    poseNetParams = PoseRegNetParams(type=11, nChan=1, wIn=128, hIn=128, batchSize=1, numJoints=14, nDims=3)
    poseNet = PoseRegNet(numpy.random.RandomState(23455), cfgParams=poseNetParams)
    poseNet.load("./NYU_network_prior.pkl")
    # comrefNetParams = ScaleNetParams(type=1, nChan=1, wIn=128, hIn=128, batchSize=1, resizeFactor=2, numJoints=1, nDims=3)
    # comrefNet = ScaleNet(numpy.random.RandomState(23455), cfgParams=comrefNetParams)
    # comrefNet.load("./net_ICVL_COM.pkl")
    comrefNetParams = ScaleNetParams(type=1, nChan=1, wIn=128, hIn=128, batchSize=1, resizeFactor=2, numJoints=1, nDims=3)
    comrefNet = ScaleNet(numpy.random.RandomState(23455), cfgParams=comrefNetParams)
    comrefNet.load("./net_NYU_COM.pkl")
    config = {'fx': 588., 'fy': 587., 'cube': (300, 300, 300)}
    # config = {'fx': 241.42, 'fy': 241.42, 'cube': (250, 250, 250)}
    # config = {'fx': 224.5, 'fy': 230.5, 'cube': (300, 300, 300)}  # Creative Gesture Camera
    # di = ICVLImporter("./capture/")
    # di.fx = 224.5
    # di.fy = 230.5
    # di.ux = 160.
    # di.uy = 120.
    rtp = RealtimeHandposePipeline(poseNet, config, di, comrefNet)

    # use filenames
    filenames = []
    for i in testSeqs[0].data:
        filenames.append(i.fileName)
    # filenames = sorted(glob.glob('./capture2/*.png'))
    rtp.processFiles(filenames)  # Threaded

    # use depth camera
    # from util.cameradevice import CreativeCameraDevice
    # dev = CreativeCameraDevice()
    # dev.start()
    # rtp.processVideo(dev)  # Threaded
    # dev.stop()
