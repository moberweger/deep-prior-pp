"""Provides importer classes for importing data from different datasets.

DepthImporter provides interface for loading the data from a dataset, esp depth images.
ICVLImporter, NYUImporter, MSRAImporter are specific instances of different importers.

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

import fnmatch
import scipy.io
import numpy as np
from PIL import Image
import glob
import os
import progressbar as pb
import struct
from data.basetypes import ICVLFrame, NamedImgSequence
from util.handdetector import HandDetector
from data.transformations import transformPoint2D
import cPickle
import matplotlib
import matplotlib.pyplot as plt


__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>, Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class DepthImporter(object):
    """
    provide baisc functionality to load depth data
    """

    def __init__(self, fx, fy, ux, uy):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x3 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[4:7, 0])), 
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[4:7, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[7:10, 0])), 
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[7:10, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[10:13, 0])), 
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[10:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:16, 0])), 
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:16, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord
        plt.show()

    @staticmethod
    def frameToPCL(frame):
        pointcloud = np.zeros((frame.dpt.shape[0] * frame.dpt.shape[1], 3), dtype=float)
        centerX = int(frame.dpt.shape[1] / 2.)
        centerY = int(frame.dpt.shape[0] / 2.)
        depth_idx = 0
        for v in range(-centerY, centerY):
            for u in range(-centerX,centerX):
                # skip invalid points
                if (frame.dpt[v+centerY,u+centerX] == 0.):
                    continue

                t = transformPoint2D([u+centerX,v+centerY],np.linalg.inv(frame.T))

                pointcloud[depth_idx,0] = float(t[0][0]-160.) * frame.dpt[v+centerY,u+centerX] / 241.42
                pointcloud[depth_idx,1] = float(t[1][0]-120.) * frame.dpt[v+centerY,u+centerX] / 241.42
                pointcloud[depth_idx,2] = frame.dpt[v+centerY,u+centerX]

                depth_idx += 1

        return pointcloud[0:depth_idx,:]


class ICVLImporter(DepthImporter):
    """
    provide functionality to load data from the ICVL dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/'):
        """
        Constructor
        :param basepath: base path of the ICVL dataset
        :return:
        """

        super(ICVLImporter, self).__init__(241.42, 241.42, 160., 120.)  # see Qian et.al.

        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 16
        
    def loadDepthMap(self,filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)  # open image
        assert len(img.getbands()) == 1  # ensure depth image
        imgdata = np.asarray(img, np.float32)

        return imgdata
        
    def loadSequence(self,seqName,subSeq = None,Nmax=float('inf'),shuffle=False,rng=None,docom=False):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param subSeq: list of subsequence names, e.g. 0, 45, 122-5
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        if (subSeq is not None) and (not isinstance(subSeq,list)):
            raise TypeError("subSeq must be None or list")

        config = {'cube':(250,250,250)}
        refineNet = None

        if subSeq is None:
            pickleCache = '{}/{}_{}_None_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, docom)
        else:
            pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, ''.join(subSeq), docom)
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache, 'rb')
                (seqName,data,config) = cPickle.load(f)
                f.close()

                # shuffle data
                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqName,data,config)

            # check for multiple subsequences
            if subSeq is not None:
                if len(subSeq) > 1:
                    missing = False
                    for i in range(len(subSeq)):
                        if not os.path.isfile('{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, subSeq[i], docom)):
                            missing = True
                            print("missing: {}".format(subSeq[i]))
                            break

                    if not missing:
                        # load first data
                        pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, subSeq[0], docom)
                        print("Loading cache data from {}".format(pickleCache))
                        f = open(pickleCache, 'rb')
                        (seqName,fullData,config) = cPickle.load(f)
                        f.close()
                        # load rest of data
                        for i in range(1,len(subSeq)):
                            pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, subSeq[i], docom)
                            print("Loading cache data from {}".format(pickleCache))
                            f = open(pickleCache, 'rb')
                            (seqName,data,config) = cPickle.load(f)
                            fullData.extend(data)
                            f.close()

                        # shuffle data
                        if shuffle and rng is not None:
                            print("Shuffling")
                            rng.shuffle(fullData)
                        if not(np.isinf(Nmax)):
                            return NamedImgSequence(seqName,fullData[0:Nmax],config)
                        else:
                            return NamedImgSequence(seqName,fullData,config)

        # Load the dataset
        objdir = '{}/Depth/'.format(self.basepath)
        trainlabels = '{}/{}.txt'.format(self.basepath, seqName)

        inputfile = open(trainlabels)
        
        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=len(inputfile.readlines()),widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()
        inputfile.seek(0)
        
        data = []
        i=0
        for line in inputfile:
            part = line.split(' ')
            #print part[0]
            # check for subsequences and skip them if necessary
            subSeqName = ''
            if subSeq is not None:
                p = part[0].split('/')
                # handle original data (unrotated '0') separately
                if ('0' in subSeq) and len(p[0])>6:
                    pass
                elif not('0' in subSeq) and len(p[0])>6:
                    i+=1
                    continue
                elif (p[0] in subSeq) and len(p[0])<=6:
                    pass
                elif not(p[0] in subSeq) and len(p[0])<=6:
                    i+=1
                    continue

                if len(p[0])<=6:
                    subSeqName = p[0]
                else:
                    subSeqName = '0'

            dptFileName = '{}/{}'.format(objdir,part[0])

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i+=1
                continue
            dpt = self.loadDepthMap(dptFileName)

            # joints in image coordinates
            gtorig = np.zeros((self.numJoints,3),np.float32)
            for joint in range(self.numJoints):
                for xyz in range(0, 3):
                    gtorig[joint,xyz] = part[joint*3+xyz+1]

            # normalized joints in 3D coordinates
            gt3Dorig = self.jointsImgTo3D(gtorig)
            #print gt3D
            #self.showAnnotatedDepth(ICVLFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dcrop,0,dptFileName,subSeqName))

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=None, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i+=1
                continue
            try:
                dpt, M, com = hd.cropArea3D(gtorig[0],size=config['cube'],docom=True)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D #normalize to com
            gtcrop = np.zeros((gtorig.shape[0],3),np.float32)
            for joint in range(gtorig.shape[0]):
                t=transformPoint2D(gtorig[joint],M)
                gtcrop[joint,0] = t[0]
                gtcrop[joint,1] = t[1]
                gtcrop[joint,2] = gtorig[joint,2]

            #print("{}".format(gt3Dorig))
            #self.showAnnotatedDepth(ICVLFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,subSeqName))

            data.append(ICVLFrame(dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,subSeqName) )
            pbar.update(i)
            i+=1

            # early stop
            if len(data)>=Nmax:
                break
                   
        inputfile.close()
        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName,data,config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName,data,config)

    def loadBaseline(self, filename, firstName=False):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        def nonblank_lines(f):
            for l in f:
                line = l.rstrip()
                if line:
                    yield line

        inputfile = open(filename)
        inputfile.seek(0)

        if firstName == True:
            off = 1
        else:
            off = 0

        data = []
        for line in nonblank_lines(inputfile):
            part = line.strip().split(' ')
            # joints in image coordinates
            ev = np.zeros((self.numJoints, 3), np.float32)
            for joint in range(ev.shape[0]):
                for xyz in range(0, 3):
                    ev[joint, xyz] = part[joint*3+xyz+off]

            gt3Dworld = self.jointsImgTo3D(ev)

            data.append(gt3Dworld)

        return data

    def loadBaseline2D(self,filename,firstName=False):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        inputfile = open(filename)
        inputfile.seek(0)

        if firstName == True:
            off = 1
        else:
            off = 0

        data = []
        for line in inputfile:
            part = line.split(' ')
            # joints in image coordinates
            ev = np.zeros((self.numJoints,2),np.float32)
            for joint in range(ev.shape[0]):
                for xyz in range(0, 2):
                    ev[joint,xyz] = part[joint*3+xyz+off]

            data.append(ev)

        return data


class NYUImporter(DepthImporter):
    """
    provide functionality to load data from the NYU hand dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/'):
        """
        Constructor
        :param basepath: base path of the ICVL dataset
        :return:
        """

        super(NYUImporter, self).__init__(588.03, 587.07, 320., 240.)

        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 36
        self.scales = {'train': 1., 'test_1': 1., 'test_2': 0.83, 'test': 1., 'train_synth': 1.,
                       'test_synth_1': 1., 'test_synth_2': 0.83, 'test_synth': 1.}
        self.restrictedJointsEval = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]

    def loadDepthMap(self,filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)

        return imgdata

    def loadSequence(self,seqName,allJoints=False,Nmax=float('inf'),shuffle=False,rng=None,docom=False):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param subSeq: list of subsequence names, e.g. 0, 45, 122-5
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        config = {'cube':(300,300,300)}
        config['cube'] = [s*self.scales[seqName] for s in config['cube']]
        refineNet = None

        pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, allJoints, docom)
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache, 'rb')
                (seqName,data,config) = cPickle.load(f)
                f.close()

                # shuffle data
                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqName,data,config)

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath,seqName)
        trainlabels = '{}/{}/joint_data.mat'.format(self.basepath, seqName)

        mat = scipy.io.loadmat(trainlabels)
        names = mat['joint_names'][0]
        joints3D = mat['joint_xyz'][0]
        joints2D = mat['joint_uvd'][0]
        if allJoints:
            eval_idxs = np.arange(36)
        else:
            eval_idxs = self.restrictedJointsEval

        self.numJoints = len(eval_idxs)

        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=joints3D.shape[0],widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        i=0
        for line in range(joints3D.shape[0]):
            dptFileName = '{0:s}/depth_1_{1:07d}.png'.format(objdir, line+1)

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)

            # joints in image coordinates
            gtorig = np.zeros((self.numJoints, 3), np.float32)
            jt = 0
            for ii in range(joints2D.shape[1]):
                if ii not in eval_idxs:
                    continue
                gtorig[jt,0] = joints2D[line,ii,0]
                gtorig[jt,1] = joints2D[line,ii,1]
                gtorig[jt,2] = joints2D[line,ii,2]
                jt += 1

            # normalized joints in 3D coordinates
            gt3Dorig = np.zeros((self.numJoints,3),np.float32)
            jt = 0
            for jj in range(joints3D.shape[1]):
                if jj not in eval_idxs:
                    continue
                gt3Dorig[jt,0] = joints3D[line,jj,0]
                gt3Dorig[jt,1] = joints3D[line,jj,1]
                gt3Dorig[jt,2] = joints3D[line,jj,2]
                jt += 1
            #print gt3D
            #self.showAnnotatedDepth(ICVLFrame(dpt,gtorig,gtorig,0,gt3Dorig,gt3Dorig,0,dptFileName,''))

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, refineNet=refineNet, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i += 1
                continue
            try:
                if allJoints:
                    dpt, M, com = hd.cropArea3D(gtorig[34], size=config['cube'],docom=True)
                else:
                    dpt, M, com = hd.cropArea3D(gtorig[13], size=config['cube'],docom=True)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = np.zeros((gtorig.shape[0], 3), np.float32)
            for joint in range(gtorig.shape[0]):
                t=transformPoint2D(gtorig[joint], M)
                gtcrop[joint, 0] = t[0]
                gtcrop[joint, 1] = t[1]
                gtcrop[joint, 2] = gtorig[joint, 2]

            #print("{}".format(gt3Dorig))
            #self.showAnnotatedDepth(ICVLFrame(dpt,gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,''))

            data.append(ICVLFrame(dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'') )
            pbar.update(i)
            i+=1

            # early stop
            if len(data)>=Nmax:
                break

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName,data,config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName,data,config)

    def loadBaseline(self, filename, gt=None):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        if gt is not None:
            mat = scipy.io.loadmat(filename)
            names = mat['conv_joint_names'][0]
            joints = mat['pred_joint_uvconf'][0]

            self.numJoints = names.shape[0]

            data = []
            for dat in range(min(joints.shape[0], gt.shape[0])):
                fname = '{0:s}/depth_1_{1:07d}.png'.format(os.path.split(filename)[0], dat+1)
                if not os.path.isfile(fname):
                    continue
                dm = self.loadDepthMap(fname)
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                jt = 0
                for i in range(joints.shape[1]):
                    if np.count_nonzero(joints[dat, i, :]) == 0:
                        continue
                    ev[jt, 0] = joints[dat, i, 0]
                    ev[jt, 1] = joints[dat, i, 1]
                    ev[jt, 2] = dm[int(ev[jt, 1]), int(ev[jt, 0])]
                    jt += 1

                for jt in range(ev.shape[0]):
                    #if ev[jt,2] == 2001. or ev[jt,2] == 0.:
                    if abs(ev[jt, 2] - gt[dat, 13, 2]) > 150.:
                        ev[jt, 2] = gt[dat, jt, 2]#np.clip(ev[jt,2],gt[dat,13,2]-150.,gt[dat,13,2]+150.) # set to groundtruth if unknown

                ev3D = self.jointsImgTo3D(ev)
                data.append(ev3D)

            return data
        else:

            def nonblank_lines(f):
                for l in f:
                    line = l.rstrip()
                    if line:
                        yield line

            inputfile = open(filename)
            # first line specifies the number of 3D joints
            self.numJoints = len(inputfile.readline().split(' ')) / 3
            inputfile.seek(0)

            data = []
            for line in nonblank_lines(inputfile):
                part = line.split(' ')
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                for joint in range(ev.shape[0]):
                    for xyz in range(0, 3):
                        ev[joint, xyz] = part[joint*3+xyz]

                gt3Dworld = self.jointsImgTo3D(ev)

                data.append(gt3Dworld)

            return data

    def loadBaseline2D(self, filename):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        mat = scipy.io.loadmat(filename)
        names = mat['conv_joint_names'][0]
        joints = mat['pred_joint_uvconf'][0]

        self.numJoints = names.shape[0]

        data = []
        for dat in range(joints.shape[0]):
            # joints in image coordinates
            ev = np.zeros((self.numJoints,2),np.float32)
            jt = 0
            for i in range(joints.shape[1]):
                if np.count_nonzero(joints[dat,i,:]) == 0:
                    continue
                ev[jt,0] = joints[dat,i,0]
                ev[jt,1] = joints[dat,i,1]
                jt += 1

            data.append(ev)

        return data

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0],3),np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f, see Thomson et al.
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,),np.float32)
        #convert to metric using f, see Thomson et.al.
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy  # TODO
        ret[2, 2] = 1
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x3 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy  # TODO
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:,0],frame.gtcrop[:,1])

        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[1::-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[1::-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[3:1:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[3:1:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[5:3:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[5:3:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[7:5:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[7:5:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[10:7:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[10:7:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[11, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[11, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[12, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[12, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = frame.dpt[row,col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        ax.format_coord = format_coord
        plt.show()

    # TODO clean interface from base class
    @staticmethod
    def frameToPCL(frame):
        pointcloud = np.zeros((frame.dpt.shape[0] * frame.dpt.shape[1], 3), dtype=float)
        centerX = int(frame.dpt.shape[1] / 2.)
        centerY = int(frame.dpt.shape[0] / 2.)
        depth_idx = 0
        for v in range(-centerY, centerY):
            for u in range(-centerX, centerX):
                # skip invalid points
                if frame.dpt[v+centerY, u+centerX] == 0.:
                    continue

                t = transformPoint2D([u+centerX, v+centerY], np.linalg.inv(frame.T))

                pointcloud[depth_idx, 0] = float(t[0][0]-320.) * frame.dpt[v+centerY, u+centerX] / 588.036885716
                pointcloud[depth_idx, 1] = float(240.-t[1][0]) * frame.dpt[v+centerY, u+centerX] / 587.075066872
                pointcloud[depth_idx, 2] = frame.dpt[v+centerY, u+centerX]

                depth_idx += 1

        return pointcloud[0:depth_idx, :]

