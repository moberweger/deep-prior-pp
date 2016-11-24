"""Provides different transformation methods on images.

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
from PIL import Image, ImageEnhance
import data.basetypes

__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>, Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


def getTransformationMatrix(center, rot, trans, scale):
    ca = numpy.cos(rot)
    sa = numpy.sin(rot)
    sc = scale
    cx = center[0]
    cy = center[1]
    tx = trans[0]
    ty = trans[1]
    t = numpy.array([ca * sc, -sa * sc, sc * (ca * (-tx - cx) + sa * ( cy + ty)) + cx,
                     sa * sc, ca * sc, sc * (ca * (-ty - cy) + sa * (-tx - cx)) + cy])
    return t


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = numpy.asmatrix(M.reshape((3, 3))) * numpy.matrix([pt[0], pt[1], 1]).T
    return numpy.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoint3D(pt, M):
    """
    Transform point in 3D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt3 = numpy.asmatrix(M.reshape((4, 4))) * numpy.matrix([pt[0], pt[1], pt[2], 1]).T
    return numpy.array([pt3[0] / pt3[3], pt3[1] / pt3[3], pt3[2] / pt3[3]])


class ImageJitterer(object):
    """
    Image jitterer than creates jittered images, using affine transformations, color, etc.
    """
    def __init__(self, rng, sz, crop=False):
        """
        Constructor
        """

        self.rng = rng
        self.imgSize = sz
        self.crop = crop

    def getJitteredParams(self, num, center=(0.0, 0.0), maxRot=(-5.0, 5.0), maxTranslate=(-2.0, 2.0),
                          maxScale=(-0.1, 0.1), mirror=True):

        if not (type(maxRot) is tuple):
            maxRot = (-maxRot, maxRot)
        if not (type(maxTranslate) is tuple):
            maxTranslate = (-maxTranslate, maxTranslate)
        if not (type(maxScale) is tuple):
            maxScale = (-maxScale, maxScale)

        alphas = self.rng.rand(num) * (maxRot[1] - maxRot[0]) + maxRot[0]
        alphas = numpy.deg2rad(alphas)

        tx = self.rng.rand(num) * (maxTranslate[1] - maxTranslate[0]) + maxTranslate[0]
        ty = self.rng.rand(num) * (maxTranslate[1] - maxTranslate[0]) + maxTranslate[0]

        sc = 2 ** -(self.rng.rand(num) * (maxScale[1] - maxScale[0]) + maxScale[0])

        if mirror:
            mi = self.rng.randint(2, size=num)  # mirror true or false
        else:
            mi = numpy.zeros(num)

        transformationMats = []
        for i in range(num):
            # First is not modified
            if i == 0:
                t = numpy.array([0, 0, 0, 1, 0])
            else:
                t = numpy.array([alphas[i], tx[i], ty[i], sc[i], mi[i]])
            transformationMats.append(t)

        return transformationMats

    def transformPoint2D(self, x_pic, y_pic, M):
        """
        Transform point
        :param x_pic:
        :param y_pic:
        :param M:
        :return:
        """

        if M.size != 6:
            raise ValueError("M not valid")

        x = M[0] * x_pic + M[1] * y_pic + M[2]
        y = M[3] * x_pic + M[4] * y_pic + M[5]
        return x, y

    def transformImg(self, img, t):
        imgT = img.transform((int(img.size[0]*t[3]),int(img.size[1]*t[3])), Image.EXTENT, (0,0,img.size[0],img.size[1]), Image.BILINEAR)
        imgT = imgT.rotate(numpy.rad2deg(t[0]), Image.BILINEAR, expand=1)
        if t[4] == 1.:
            imgT = imgT.transpose(Image.FLIP_LEFT_RIGHT)

        # crop only valid part
        if self.crop:
            imgT = imgT.crop(self.getInscribedRectangle(t[0], (img.size[0]*t[3], img.size[1]*t[3])))

        # crop from translation
        imgT = imgT.resize((int(self.imgSize[0]*1.1), int(self.imgSize[1]*1.1)), Image.BILINEAR)
        xstart = int((imgT.size[0] // 2 - t[1]) - self.imgSize[0] // 2)
        ystart = int((imgT.size[1] // 2 - t[2]) - self.imgSize[1] // 2)
        assert xstart >= 0 and ystart >= 0
        return imgT.crop((xstart, ystart, xstart+self.imgSize[0], ystart+self.imgSize[1]))

    def getJitteredImgs(self, img, num, maxRot=(-5.0, 5.0), maxTranslate=(-2.0, 2.0), maxScale=(-0.1, 0.1), augmentColor=False):
        """ 
        Take img and jitter it
        :return: a list of all jittered images
        """

        cx = img.size[0] / 2
        cy = img.size[1] / 2

        tMats = self.getJitteredParams(center=(cx, cy), num=num, maxRot=maxRot, maxTranslate=maxTranslate,
                                       maxScale=maxScale)
        imgs = []
        for i in range(len(tMats)):
            t = tMats[i]
            imgT = self.transformImg(img, t)

            if augmentColor:
                # jitter colors
                color = ImageEnhance.Color(imgT)
                imgT = color.enhance(self.rng.uniform(0.7, 1))

                # jitter contrast
                contr = ImageEnhance.Contrast(imgT)
                imgT = contr.enhance(self.rng.uniform(0.7, 1))

                # jitter brightness
                bright = ImageEnhance.Brightness(imgT)
                imgT = bright.enhance(self.rng.uniform(0.7, 1))

                # add noise
                im = numpy.asarray(imgT).astype('int') + numpy.rint(self.rng.normal(0, 4, numpy.asarray(imgT).shape)).astype('int')
                im = numpy.clip(im, 0, 255).astype('uint8')
                imgT = Image.fromarray(im)

            # add image
            imgs.append(imgT)

        return imgs, tMats

    def applyJitterImg(self, img, tMats):
        imgs = []
        for i in range(len(tMats)):
            t = tMats[i]
            imgT = self.transformImg(img, t)

            # add image
            imgs.append(imgT)

        return imgs

    def getJitteredImgSeq(self, imgSeq, num, maxRot=(-5.0, 5.0), maxTranslate=(-2.0, 2.0), maxScale=(-0.1, 0.1)):
        """
        Take every img in the sequence (ie. list of Frames) and jitter it
        return a list of all jittered 
        
        :param imgSeq: list of Frames
        """

        seq = []
        for frame in imgSeq:
            imgs = self.getJitteredImgs(frame.img, num, maxRot, maxTranslate, maxScale)
            for i in range(num):
                seq.append(data.basetypes.Frame(imgs[i], frame.dpt, frame.rot, frame.tra, frame.className))
        return seq

    def maximumInscribedRectangle(self, mask):
        """
        http://www.imagingshop.com/articles/automatic-cropping-non-rectangular-images
        This is a very slow, genearal purpose, enumerative approach
        :param mask: image mask
        :return: largest inscribed rectangle within mask
        """

        def getSize(ww, hh):
            return ww * hh

        (height, width) = mask.size

        squares = numpy.zeros((height, width), dtype=int)

        # process bottom boundary of the mask
        row = (height - 1)

        for col in range(0, width):
            if mask.getpixel((row, col)):
                squares[row, col] = 1

        # process right boundary of the mask
        col = (width - 1)

        for row in range(0, height):
            if mask.getpixel((row, col)):
                squares[row, col] = 1

        # process internal pixels of the mask
        for row in range(height - 2, -1, -1):
            for col in range(width - 2, -1, -1):
                if mask.getpixel((row, col)):
                    a = squares[row, col + 1]
                    b = squares[row + 1, col]
                    c = squares[row + 1, col + 1]
                    squares[row, col] = (min(min(a, b), c) + 1)

        sizes = numpy.zeros((height, width), dtype=int)

        maxSquare = 0

        for row in range(0, height):
            for col in range(0, width):
                square = squares[row, col]
                sizes[row, col] = getSize(square, square)

                if square > maxSquare:
                    maxSquare = square

        # find largest rectangles with width >= height
        height2width = [None]*(maxSquare + 1)

        widths = numpy.zeros((height, width), dtype=int)
        heights = numpy.zeros((height, width), dtype=int)

        for row in range(0, height):
            for s in range(0, maxSquare+1):
                height2width[s] = 0

            for col in range(width - 1, -1, -1):
                square = squares[row, col]

                if square > 0:
                    maxSize = sizes[row, col]

                    for rectHeight in range(square, 0, -1):
                        rectWidth = height2width[rectHeight]
                        rectWidth = max(rectWidth + 1, square)
                        height2width[rectHeight] = rectWidth
                        size = getSize(rectWidth, rectHeight)
                        if size >= maxSize:
                            maxSize = size
                            widths[row, col] = rectWidth
                            heights[row, col] = rectHeight

                    sizes[row, col] = maxSize

                for s in range(square + 1, maxSquare+1):
                    # widths larger that 'square' will not be available
                    height2width[s] = 0

        # find largest rectangles with width < height
        width2height = [None]*(maxSquare + 1)

        for col in range(0, width):
            for s in range(0, maxSquare+1):
                width2height[s] = 0

            for row in range(height - 1, -1, -1):
                square = squares[row, col]

                if square > 0:
                    maxSize = sizes[row, col]

                    for rectWidth in range(square, rectWidth, -1):
                        rectHeight = width2height[rectWidth]
                        rectHeight = max(rectHeight + 1, square)
                        width2height[rectWidth] = rectHeight
                        size = getSize(rectWidth, rectHeight)

                        if size > maxSize:
                            maxSize = size
                            widths[row, col] = rectWidth
                            heights[row, col] = rectHeight

                    sizes[row, col] = maxSize

                for s in range(square + 1, maxSquare+1):
                    # heights larger that 'square' will not be available
                    width2height[s] = 0

        # find the largest rectangle
        maxSize = 0
        rectWidth = 0
        rectHeight = 0
        rectRow = 0
        rectCol = 0

        for row in range(0, height):
            for col in range(0, width):
                size = sizes[row, col]
                if size > maxSize:
                    maxSize = size
                    rectRow = row
                    rectCol = col
                    rectWidth = widths[row, col]
                    rectHeight = heights[row, col]

        return (rectCol, rectRow, rectCol + rectWidth, rectRow + rectHeight)

    def getInscribedRectangle(self, angle, rectSz):
        """
        From https://stackoverflow.com/questions/5789239/calculate-largest-rectangle-in-a-rotated-rectangle
        :param angle: angle in radians
        :param rectSz: rectangle size
        :return:
        """

        imgSzw = rectSz[0]
        imgSzh = rectSz[1]

        quadrant = int(numpy.floor(angle / (numpy.pi / 2.))) & 3
        sign_alpha = angle if (quadrant & 1) == 0 else numpy.pi - angle
        alpha = (sign_alpha % numpy.pi + numpy.pi) % numpy.pi

        bbw = imgSzw * numpy.cos(alpha) + imgSzh * numpy.sin(alpha)
        bbh = imgSzw * numpy.sin(alpha) + imgSzh * numpy.cos(alpha)

        gamma = numpy.arctan2(bbw, bbh) if imgSzw < imgSzh else numpy.arctan2(bbh, bbw)
        delta = numpy.pi - alpha - gamma

        length = imgSzh if imgSzw < imgSzh else imgSzw
        d = length * numpy.cos(alpha)
        a = d * numpy.sin(alpha) / numpy.sin(delta)

        y = a * numpy.cos(gamma)
        x = y * numpy.tan(gamma)

        return (int(x), int(y), int(x + bbw - 2*x), int(y + bbh - 2*y))