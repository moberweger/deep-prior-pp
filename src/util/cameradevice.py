"""Basis for depth camera devices.

CameraDevice provides interface for managing depth cameras.
It can be used to retrieve basic information and read
depth and color frames.

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

import time
import numpy
import cv2
import scipy.misc
import lib_dscapture as dsc
import openni

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class CameraDevice(object):
    """
    Abstract class that handles all camera devices
    """

    def __init__(self, mirror=False):
        """
        Initialize device
        :param mirror: mirror all images
        :return: None
        """

        self.mirror = mirror

    def start(self):
        """
        Start device
        :return: None
        """
        raise NotImplementedError("!")

    def stop(self):
        """
        Stop device
        :return: None
        """
        raise NotImplementedError("!")

    def saveDepth(self, data, file_name):
        """
        Save data to file, we need special treatment because we have 16bit depth
        :param data: data
        :param file_name: file name
        :return: None
        """

        im = scipy.misc.toimage(data.astype('uint16'), high=numpy.max(data), low=numpy.min(data), mode='I')
        im.save(file_name+'.png')
        # read with: b = scipy.misc.imread('my16bit.png')

    def saveRGB(self, data, file_name):
        """
        Save data to file 3x8bit color
        :param data: data
        :param file_name: file name
        :return: None
        """

        assert len(data.shape) == 3

        scipy.misc.imsave(file_name+'.png', data)
        # read with: b = scipy.misc.imread('my.png')

    def getDepth(self):
        """
        Return a median smoothed depth image
        :return: depth data as numpy array
        """
        raise NotImplementedError("!")

    def getRGB(self):
        """
        Return a bit color image
        :return: color image as numpy array
        """
        raise NotImplementedError("!")

    def getGrayScale(self):
        """
        Return a grayscale image
        :return: grayscale image as numpy array
        """
        raise NotImplementedError("!")

    def getRGBD(self):
        """
        Return a color + depth image
        :return: RGB-D image as 4-channel numpy array
        """

        ret_rgb, c = self.getRGB()
        ret_d, d = self.getDepth()

        return ret_rgb and ret_d, c.astype('float32'), d.astype('float32')

    def getLastColorNum(self):
        """
        Get frame number of last color frame
        :return: frame number
        """
        raise NotImplementedError("!")

    def getLastDepthNum(self):
        """
        Get frame number of last depth frame
        :return: frame number
        """
        raise NotImplementedError("!")

    def getDepthIntrinsics(self):
        """
        Get intrinsic matrix of depth camera
        :return: 3x3 intrinsic camera matrix
        """
        raise NotImplementedError("!")

    def getColorIntrinsics(self):
        """
        Get intrinsic matrix of color camera
        :return: 3x3 intrinsic camera matrix
        """
        raise NotImplementedError("!")

    def getExtrinsics(self):
        """
        Get extrinsic matrix from color to depth camera
        :return: 4x3 extrinsic camera matrix
        """
        raise NotImplementedError("!")


class CreativeCameraDevice(CameraDevice):
    """ DepthSense camera class, for Creative Gesture Camera, DS325, etc."""

    def __init__(self, mirror=False):
        """
        Initialize device
        :param mirror: mirror image
        """

        super(CreativeCameraDevice, self).__init__(mirror)

    def start(self):
        """
        Start device
        :return: None
        """
        dsc.start()

    def stop(self):
        """
        Stop device
        :return: None
        """
        dsc.stop()

    def getDepth(self):
        """
        Return a median smoothed depth image
        :return: depth data as numpy array
        """

        if self.mirror:
            depth = dsc.getDepthMap()[:, ::-1]
        else:
            depth = dsc.getDepthMap()
        depth = cv2.medianBlur(depth, 3)
        return (numpy.count_nonzero(depth) != 0), numpy.asarray(depth, numpy.float32)

    def getRGB(self):
        """
        Return a bit color image
        :return: color image as numpy array
        """

        if self.mirror:
            image = dsc.getColourMap()[:, ::-1, :]
        else:
            image = dsc.getColourMap()
        return (numpy.count_nonzero(image) != 0), image

    def getGrayScale(self):
        """
        Return a grayscale image
        :return: grayscale image as numpy array
        """

        if self.mirror:
            image = dsc.getColorMap()[:, ::-1, :]
        else:
            image = dsc.getColorMap()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (numpy.count_nonzero(grey) != 0), grey.transpose()

    def getLastColorNum(self):
        """
        Get frame number of last color frame
        :return: frame number
        """
        return dsc.getLastColorNum()

    def getLastDepthNum(self):
        """
        Get frame number of last depth frame
        :return: frame number
        """
        return dsc.getLastDepthNum()

    def getDepthIntrinsics(self):
        """
        Get intrinsic matrix of depth camera
        :return: 3x3 intrinsic camera matrix
        """

        return dsc.getDepthIntrinsics()

    def getColorIntrinsics(self):
        """
        Get intrinsic matrix of color camera
        :return: 3x3 intrinsic camera matrix
        """
        return dsc.getColorIntrinsics()

    def getExtrinsics(self):
        """
        Get extrinsic matrix from color to depth camera
        :return: 4x3 extrinsic camera matrix
        """
        return dsc.getExtrinsics()


class DepthSenseCameraDevice(CameraDevice):
    """
    Class for OpenNI based devices, e.g. Kinect, Asus Xtion
    """

    def __init__(self, mirror=False):
        """
        Initialize device
        :param mirror: mirror image
        """

        super(DepthSenseCameraDevice, self).__init__(mirror)

    def start(self):
        """
        Stop device
        :return: None
        """
        self.ctx = openni.Context()
        self.ctx.init()

        # Create a depth generator
        self.depth = openni.DepthGenerator()
        self.depth.create(self.ctx)

        # Set it to VGA maps at 30 FPS
        self.depth.set_resolution_preset(openni.RES_VGA)
        self.depth.fps = 30

        # Create a color generator
        self.color = openni.ImageGenerator()
        self.color.create(self.ctx)

        # Set it to VGA maps at 30 FPS
        self.color.set_resolution_preset(openni.RES_VGA)
        self.color.fps = 30

        # Start generating
        self.ctx.start_generating_all()

    def stop(self):
        """
        Stop device
        :return: None
        """

        self.ctx.stop_generating_all()
        self.ctx.shutdown()

    def getDepth(self):
        """
        Return a median smoothed depth image
        :return: depth data as numpy array
        """

        # Get the pixel at these coordinates
        try:
            # Wait for new data to be available
            self.ctx.wait_one_update_all(self.depth)
        except openni.OpenNIError, err:
            print "Failed updating data:", err
        else:
            dpt = numpy.asarray(self.depth.get_tuple_depth_map(), dtype='float32').reshape(self.depth.map.height, self.depth.map.width)

            return True, dpt

    def getRGB(self):
        """
        Return a median smoothed depth image
        :return: depth data as numpy array
        """

        # Get the pixel at these coordinates
        try:
            # Wait for new data to be available
            self.ctx.wait_one_update_all(self.color)
        except openni.OpenNIError, err:
            print "Failed updating data:", err
        else:
            dpt = numpy.asarray(self.color.get_tuple_depth_map(), dtype='float32').reshape(self.color.map.height, self.color.map.width)

            return True, dpt


class FileDevice(CameraDevice):
    """
    Fake class to load images from file
    """

    def __init__(self, filenames, importer, mirror=False):
        """
        Initialize device
        :param mirror: mirror all images
        :return: None
        """

        super(FileDevice, self).__init__(mirror)

        if not isinstance(filenames, list):
            raise ValueError("Files must be list of filenames.")

        self.filenames = filenames
        self.importer = importer
        self.depth_intrinsics = importer.getCameraIntrinsics()
        self.color_intrinsics = numpy.zeros((3, 3))
        self.extrinsics = numpy.zeros((3, 4))
        self.mirror = mirror

        self.last_color_num = 0
        self.last_depth_num = 0

    def start(self):
        """
        Start device
        :return: None
        """
        pass

    def stop(self):
        """
        Stop device
        :return: None
        """
        pass

    def getDepth(self):
        """
        Return a median smoothed depth image
        :return: depth data as numpy array
        """
        time.sleep(0.01)
        frame = self.importer.loadDepthMap(self.filenames[self.last_depth_num])
        self.last_depth_num += 1
        return True, frame

    def getRGB(self):
        """
        Return a bit color image
        :return: color image as numpy array
        """
        raise NotImplementedError("!")

    def getGrayScale(self):
        """
        Return a grayscale image
        :return: grayscale image as numpy array
        """
        raise NotImplementedError("!")

    def getRGBD(self):
        """
        Return a color + depth image
        :return: RGB-D image as 4-channel numpy array
        """

        ret_rgb, c = self.getRGB()
        ret_d, d = self.getDepth()

        return ret_rgb and ret_d, c.astype('float32'), d.astype('float32')

    def getLastColorNum(self):
        """
        Get frame number of last color frame
        :return: frame number
        """
        return self.last_color_num

    def getLastDepthNum(self):
        """
        Get frame number of last depth frame
        :return: frame number
        """
        return self.last_depth_num

    def getDepthIntrinsics(self):
        """
        Get intrinsic matrix of depth camera
        :return: 3x3 intrinsic camera matrix
        """
        return self.depth_intrinsics

    def getColorIntrinsics(self):
        """
        Get intrinsic matrix of color camera
        :return: 3x3 intrinsic camera matrix
        """
        return self.color_intrinsics

    def getExtrinsics(self):
        """
        Get extrinsic matrix from color to depth camera
        :return: 4x3 extrinsic camera matrix
        """
        return self.extrinsics
