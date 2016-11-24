"""Provides a basic hand detector in depth images.

HandDetector provides interface for detecting hands in depth image, by using the center of mass.

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
import os
import cv2
from scipy import stats, ndimage

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandDetector(object):
    """
    Detect hand based on simple heuristic, centered at Center of Mass
    """

    RESIZE_BILINEAR = 0
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self, dpt, fx, fy, importer=None, refineNet=None):
        """
        Constructor
        :param dpt: depth image
        :param fx: camera focal lenght
        :param fy: camera focal lenght
        """
        self.dpt = dpt
        self.maxDepth = min(1500, dpt.max())
        self.minDepth = max(10, dpt.min())
        # set values out of range to 0
        self.dpt[self.dpt > self.maxDepth] = 0.
        self.dpt[self.dpt < self.minDepth] = 0.
        # camera settings
        self.fx = fx
        self.fy = fy
        # Optional refinement of CoM
        self.refineNet = refineNet
        self.importer = importer
        # depth resize method
        self.resizeMethod = self.RESIZE_CV2_NN

    def calculateCoM(self, dpt):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < self.minDepth] = 0
        dc[dc > self.maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = numpy.count_nonzero(dc)
        com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

        if num == 0:
            return numpy.array((0, 0, 0), numpy.float)
        else:
            return com/num

    def checkImage(self, tol):
        """
        Check if there is some content in the image
        :param tol: tolerance
        :return:True if image is contentful, otherwise false
        """
        # print numpy.std(self.dpt)
        if numpy.std(self.dpt) < tol:
            return False
        else:
            return True

    def getNDValue(self):
        """
        Get value of not defined depth value distances
        :return:value of not defined depth value
        """
        if self.dpt[self.dpt < self.minDepth].shape[0] > self.dpt[self.dpt > self.maxDepth].shape[0]:
            return stats.mode(self.dpt[self.dpt < self.minDepth])[0][0]
        else:
            return stats.mode(self.dpt[self.dpt > self.maxDepth])[0][0]

    @staticmethod
    def bilinearResize(src, dsize, ndValue):
        """
        Bilinear resizing with sparing out not defined parts of the depth map
        :param src: source depth map
        :param dsize: new size of resized depth map
        :param ndValue: value of not defined depth
        :return:resized depth map
        """

        dst = numpy.zeros((dsize[1], dsize[0]), dtype=numpy.float32)

        x_ratio = float(src.shape[1] - 1) / dst.shape[1]
        y_ratio = float(src.shape[0] - 1) / dst.shape[0]
        for row in range(dst.shape[0]):
            y = int(row * y_ratio)
            y_diff = (row * y_ratio) - y  # distance of the nearest pixel(y axis)
            y_diff_2 = 1 - y_diff
            for col in range(dst.shape[1]):
                x = int(col * x_ratio)
                x_diff = (col * x_ratio) - x  # distance of the nearest pixel(x axis)
                x_diff_2 = 1 - x_diff
                y2_cross_x2 = y_diff_2 * x_diff_2
                y2_cross_x = y_diff_2 * x_diff
                y_cross_x2 = y_diff * x_diff_2
                y_cross_x = y_diff * x_diff

                # mathematically impossible, but just to be sure...
                if(x+1 >= src.shape[1]) | (y+1 >= src.shape[0]):
                    raise UserWarning("Shape mismatch")

                # set value to ND if there are more than two values ND
                numND = int(src[y, x] == ndValue) + int(src[y, x + 1] == ndValue) + int(src[y + 1, x] == ndValue) + int(
                    src[y + 1, x + 1] == ndValue)
                if numND > 2:
                    dst[row, col] = ndValue
                    continue
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x
                # interpolate only over known values, switch to linear interpolation
                if src[y, x] == ndValue:
                    y2_cross_x2 = 0.
                    y2_cross_x = 1. - y_cross_x - y_cross_x2
                if src[y, x + 1] == ndValue:
                    y2_cross_x = 0.
                    if y2_cross_x2 != 0.:
                        y2_cross_x2 = 1. - y_cross_x - y_cross_x2
                if src[y + 1, x] == ndValue:
                    y_cross_x2 = 0.
                    y_cross_x = 1. - y2_cross_x - y2_cross_x2
                if src[y + 1, x + 1] == ndValue:
                    y_cross_x = 0.
                    if y_cross_x2 != 0.:
                        y_cross_x2 = 1. - y2_cross_x - y2_cross_x2

                # print src[y, x], src[y, x+1],src[y+1, x],src[y+1, x+1]
                # normalize weights
                if not ((y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.)):
                    sc = 1. / (y_cross_x + y_cross_x2 + y2_cross_x + y2_cross_x2)
                    y2_cross_x2 *= sc
                    y2_cross_x *= sc
                    y_cross_x2 *= sc
                    y_cross_x *= sc
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x

                if (y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.):
                    dst[row, col] = ndValue
                else:
                    dst[row, col] = y2_cross_x2 * src[y, x] + y2_cross_x * src[y, x + 1] + y_cross_x2 * src[
                        y + 1, x] + y_cross_x * src[y + 1, x + 1]

        return dst

    def comToBounds(self, com, size):
        """
        Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: xstart, xend, ystart, yend, zstart, zend
        """
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(numpy.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2]*self.fx))
        xend = int(numpy.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2]*self.fx))
        ystart = int(numpy.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2]*self.fy))
        yend = int(numpy.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2]*self.fy))
        return xstart, xend, ystart, yend, zstart, zend

    def getCrop(self, dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        """
        Crop patch from image
        :param dpt: depth image to crop from
        :param xstart: start x
        :param xend: end x
        :param ystart: start y
        :param yend: end y
        :param zstart: start z
        :param zend: end z
        :param thresh_z: threshold z values
        :return: cropped image
        """
        if len(dpt.shape) == 2:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1]))), mode='constant', constant_values=0)
        elif len(dpt.shape) == 3:
            cropped = dpt[max(ystart, 0):min(yend, dpt.shape[0]), max(xstart, 0):min(xend, dpt.shape[1]), :].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0),
                                           abs(yend)-min(yend, dpt.shape[0])),
                                          (abs(xstart)-max(xstart, 0),
                                           abs(xend)-min(xend, dpt.shape[1])),
                                          (0, 0)), mode='constant', constant_values=0)
        else:
            raise NotImplementedError()

        if thresh_z is True:
            msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
            msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later
        return cropped

    def resizeCrop(self, crop, sz):
        """
        Resize cropped image
        :param crop: crop
        :param sz: size
        :return: resized image
        """
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_BILINEAR:
            rz = self.bilinearResize(crop, sz, self.getNDValue())
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(crop, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")
        return rz

    def applyCrop3D(self, dpt, com, size, dsize, thresh_z=True, background=None):

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z)

        wb = (xend - xstart)
        hb = (yend - ystart)
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # depth resize
        rz = self.resizeCrop(cropped, sz)

        if background is None:
            background = self.getNDValue()  # use background as filler
        ret = numpy.ones(dsize, numpy.float32) * background
        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz

        return ret

    def cropArea3D(self, com=None, size=(250, 250, 250), dsize=(128, 128), docom=False):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """

        # print com, self.importer.jointImgTo3D(com)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(self.dpt, cmap=matplotlib.cm.jet)

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        if com is None:
            com = self.calculateCoM(self.dpt)

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)
        # ax.plot(com[0],com[1],marker='.')

        #############
        # for simulating COM within cube
        if docom is True:
            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                if numpy.isclose(com[2], 0):
                    com[2] = 300.
            com[0] += xstart
            com[1] += ystart

            # calculate boundaries
            xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

            # crop patch from source
            cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)
        # ax.plot(com[0],com[1],marker='x')

        ##############
        if self.refineNet is not None and self.importer is not None:
            rz = self.resizeCrop(cropped, dsize)
            newCom3D = self.refineCoM(rz, size, com) + self.importer.jointImgTo3D(com)
            com = self.importer.joint3DToImg(newCom3D)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]

            # calculate boundaries
            xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

            # crop patch from source
            cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)

        # ax.plot(com[0],com[1],marker='o')
        # plt.show(block=True)

        #############

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = numpy.asmatrix(numpy.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        if wb > hb:
            sz = (dsize[0], hb * dsize[0] / wb)
        else:
            sz = (wb * dsize[1] / hb, dsize[1])

        # print com, sz, cropped.shape, xstart, xend, ystart, yend, hb, wb, zstart, zend
        if cropped.shape[0] > cropped.shape[1]:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[1] / float(cropped.shape[0]))
        else:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[0] / float(cropped.shape[1]))
        scale[2, 2] = 1

        # depth resize
        rz = self.resizeCrop(cropped, sz)

        # pylab.imshow(rz); pylab.gray();t=transformPoint2D(com,scale*trans);pylab.scatter(t[0],t[1]); pylab.show()
        ret = numpy.ones(dsize, numpy.float32) * self.getNDValue()  # use background as filler
        xstart = int(numpy.floor(dsize[0] / 2. - rz.shape[1] / 2.))
        xend = int(xstart + rz.shape[1])
        ystart = int(numpy.floor(dsize[1] / 2. - rz.shape[0] / 2.))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape
        off = numpy.asmatrix(numpy.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        # fig = plt.figure()
        # ax = fig.add_subplot(131)
        # ax.imshow(cropped, cmap=matplotlib.cm.jet)
        # ax = fig.add_subplot(132)
        # ax.imshow(rz, cmap=matplotlib.cm.jet)
        # ax = fig.add_subplot(133)
        # ax.imshow(ret, cmap=matplotlib.cm.jet)
        # plt.show(block=False)

        # print trans,scale,off,off*scale*trans
        return ret, off * scale * trans, com

    def checkPose(self, joints):
        """
        Check if pose is anatomically possible
        @see Serre: Kinematic model of the hand using computer vision
        :param joints: joint locations R^16x3
        :return: true if pose is possible
        """

        # check dip, pip of fingers

        return True

    def track(self, com, size=(250, 250, 250), dsize=(128, 128), doHandSize=True):
        """
        Detect the hand as closest object to camera
        :param size: bounding box size
        :return: center of mass of hand
        """

        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

        # crop patch from source
        cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)

        # predict movement of CoM
        if self.refineNet is not None and self.importer is not None:
            rz = self.resizeCrop(cropped, dsize)
            newCom3D = self.refineCoM(rz, size, com) + self.importer.jointImgTo3D(com)
            com = self.importer.joint3DToImg(newCom3D)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
        else:
            raise RuntimeError("Need refineNet for this")

        if doHandSize is True:
            # refined contour for size estimation
            zstart = com[2] - size[2] / 2.
            zend = com[2] + size[2] / 2.
            part_ref = self.dpt.copy()
            part_ref[part_ref < zstart] = 0
            part_ref[part_ref > zend] = 0
            part_ref[part_ref != 0] = 10  # set to something
            ret, thresh_ref = cv2.threshold(part_ref, 1, 255, cv2.THRESH_BINARY)
            contours_ref, _ = cv2.findContours(thresh_ref.astype(dtype=numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find the largest contour
            areas = [cv2.contourArea(cc) for cc in contours_ref]
            c_max = numpy.argmax(areas)

            # final result
            return com, self.estimateHandsize(contours_ref[c_max], com, size)
        else:
            return com, size

    def refineCoMIterative(self, com, num_iter, size=(250, 250, 250)):
        """
        Refine com iteratively
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param num_iter: number of iterations
        :param size: (x,y,z) extent of the source crop volume in mm
        :return: refined com
        """
        for k in xrange(num_iter):
            # calculate boundaries
            xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com, size)

            # crop
            cropped = self.getCrop(self.dpt, xstart, xend, ystart, yend, zstart, zend)

            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
            com[0] += max(xstart, 0)
            com[1] += max(ystart, 0)

        return com

    def detect(self, size=(250, 250, 250), doHandSize=True):
        """
        Detect the hand as closest object to camera
        :param size: bounding box size
        :return: center of mass of hand
        """

        steps = 20
        dz = (self.maxDepth - self.minDepth)/float(steps)
        for i in range(steps):
            part = self.dpt.copy()
            part[part < i*dz + self.minDepth] = 0
            part[part > (i+1)*dz + self.minDepth] = 0
            part[part != 0] = 10  # set to something
            ret, thresh = cv2.threshold(part, 1, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype(dtype=numpy.uint8)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in range(len(contours)):
                if cv2.contourArea(contours[c]) > 200:

                    # centroid
                    M = cv2.moments(contours[c])
                    cx = int(numpy.rint(M['m10']/M['m00']))
                    cy = int(numpy.rint(M['m01']/M['m00']))

                    # crop
                    xstart = int(max(cx-100, 0))
                    xend = int(min(cx+100, self.dpt.shape[1]-1))
                    ystart = int(max(cy-100, 0))
                    yend = int(min(cy+100, self.dpt.shape[0]-1))

                    cropped = self.dpt[ystart:yend, xstart:xend].copy()
                    cropped[cropped < i*dz + self.minDepth] = 0.
                    cropped[cropped > (i+1)*dz + self.minDepth] = 0.
                    com = self.calculateCoM(cropped)
                    if numpy.allclose(com, 0.):
                        com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
                    com[0] += xstart
                    com[1] += ystart

                    # refine iteratively
                    com = self.refineCoMIterative(com, 5, size)
                    zstart = com[2] - size[2] / 2.
                    zend = com[2] + size[2] / 2.

                    if doHandSize is True:
                        # refined contour for size estimation
                        part_ref = self.dpt.copy()
                        part_ref[part_ref < zstart] = 0
                        part_ref[part_ref > zend] = 0
                        part_ref[part_ref != 0] = 10  # set to something
                        ret, thresh_ref = cv2.threshold(part_ref, 1, 255, cv2.THRESH_BINARY)
                        contours_ref, _ = cv2.findContours(thresh_ref.astype(dtype=numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        # find the largest contour
                        areas = [cv2.contourArea(cc) for cc in contours_ref]
                        c_max = numpy.argmax(areas)

                        # final result
                        return com, self.estimateHandsize(contours_ref[c_max], com, size)
                    else:
                        return com, size

        # no appropriate hand detected
        return numpy.array((0, 0, 0), numpy.float), size

    def refineCoM(self, cropped, size, com):
        """
        Refines the detection result of the hand
        :return: center of hand
        """

        imgD = numpy.asarray(cropped.copy(), 'float32')

        imgD[imgD == 0] = com[2] + (size[2] / 2.)
        imgD[imgD >= com[2] + (size[2] / 2.)] = com[2] + (size[2] / 2.)
        imgD[imgD <= com[2] - (size[2] / 2.)] = com[2] - (size[2] / 2.)
        imgD -= com[2]
        imgD /= (size[2] / 2.)

        test_data = numpy.zeros((1, 1, cropped.shape[0], cropped.shape[1]), dtype='float32')
        test_data[0, 0] = imgD
        # test_data2 = numpy.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2]//2, test_data.shape[3]//2), dtype='float32')
        # test_data4 = numpy.zeros((test_data2.shape[0], test_data2.shape[1], test_data2.shape[2]//2, test_data2.shape[3]//2), dtype='float32')
        # for j in range(test_data.shape[0]):
        #     for i in range(test_data.shape[1]):
        #         test_data2[j, i, :, :] = cv2.resize(test_data[j, i, :, :], (test_data2.shape[3], test_data2.shape[2]))
        #         test_data4[j, i, :, :] = cv2.resize(test_data2[j, i, :, :], (test_data4.shape[3], test_data4.shape[2]))

        dsize = (int(test_data.shape[2]//2), int(test_data.shape[3]//2))
        xstart = int(test_data.shape[2]/2-dsize[0]/2)
        xend = xstart + dsize[0]
        ystart = int(test_data.shape[3]/2-dsize[1]/2)
        yend = ystart + dsize[1]
        test_data2 = test_data[:, :, ystart:yend, xstart:xend]

        dsize = (int(test_data.shape[2]//4), int(test_data.shape[3]//4))
        xstart = int(test_data.shape[2]/2-dsize[0]/2)
        xend = xstart + dsize[0]
        ystart = int(test_data.shape[3]/2-dsize[1]/2)
        yend = ystart + dsize[1]
        test_data4 = test_data[:, :, ystart:yend, xstart:xend]

        jts = self.refineNet.computeOutput([test_data, test_data2, test_data4])
        return jts[0]*(size[2]/2.)

    def estimateHandsize(self, contours, com, cube=(250, 250, 250), tol=0):
        """
        Estimate hand size from depth image
        :param contours: contours of hand
        :param com: center of mass
        :param cube: default cube
        :param tol: tolerance to be added to all sides
        :return: metric cube for cropping (x, y, z)
        """
        x, y, w, h = cv2.boundingRect(contours)

        # drawing = numpy.zeros((480, 640), dtype=float)
        # cv2.drawContours(drawing, [contours], 0, (255, 0, 244), 1, 8)
        # cv2.rectangle(drawing, (x, y), (x+w, y+h), (244, 0, 233), 2, 8, 0)
        # cv2.imshow("contour", drawing)

        # convert to cube
        xstart = (com[0] - w / 2.) * com[2] / self.fx
        xend = (com[0] + w / 2.) * com[2] / self.fx
        ystart = (com[1] - h / 2.) * com[2] / self.fy
        yend = (com[1] + h / 2.) * com[2] / self.fy
        szx = xend - xstart
        szy = yend - ystart
        sz = (szx + szy) / 2.
        cube = (sz + tol, sz + tol, sz + tol)

        return cube
