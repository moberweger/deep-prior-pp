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
    pt2 = numpy.dot(numpy.asarray(M).reshape((3, 3)), numpy.asarray([pt[0], pt[1], 1]))
    return numpy.asarray([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoints2D(pts, M):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param M: transformation matrix
    :return: transformed points
    """
    ret = pts.copy()
    for i in range(pts.shape[0]):
        ret[i, 0:2] = transformPoint2D(pts[i, 0:2], M)
    return ret


def rotatePoint2D(p1, center, angle):
    """
    Rotate a point in 2D around center
    :param p1: point in 2D (u,v,d)
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated point
    """
    alpha = angle * numpy.pi / 180.
    pp = p1.copy()
    pp[0:2] -= center[0:2]
    pr = numpy.zeros_like(pp)
    pr[0] = pp[0]*numpy.cos(alpha) - pp[1]*numpy.sin(alpha)
    pr[1] = pp[0]*numpy.sin(alpha) + pp[1]*numpy.cos(alpha)
    pr[2] = pp[2]
    ps = pr
    ps[0:2] += center[0:2]
    return ps


def rotatePoints2D(pts, center, angle):
    """
    Transform points in 2D coordinates
    :param pts: point coordinates
    :param center: 2D center of rotation
    :param angle: angle in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in xrange(pts.shape[0]):
        ret[i] = rotatePoint2D(pts[i], center, angle)
    return ret


def getRotationMatrix(angle_x, angle_y, angle_z):
    """
    Get rotation matrix
    :param angle_x: angle around x-axis in deg
    :param angle_y: angle around y-axis in deg
    :param angle_z: angle around z-axis in deg
    :return: 4x4 rotation matrix
    """
    alpha_x = angle_x * numpy.pi / 180.
    alpha_y = angle_y * numpy.pi / 180.
    alpha_z = angle_z * numpy.pi / 180.
    R = numpy.eye(4)
    from transforms3d.euler import euler2mat
    R[:3, :3] = euler2mat(alpha_x, alpha_y, alpha_z, 'rxyz')
    return R


def rotatePoint3D(p1, center, angle_x, angle_y, angle_z):
    """
    Rotate a point in 3D around center
    :param p1: point in 3D (x,y,z)
    :param center: 3D center of rotation
    :param angle_x: angle around x-axis in deg
    :param angle_y: angle around y-axis in deg
    :param angle_z: angle around z-axis in deg
    :return: rotated point
    """
    pp = p1.copy()
    pp -= center
    R = getRotationMatrix(angle_x, angle_y, angle_z)
    pr = numpy.array([pp[0], pp[1], pp[2], 1])
    ps = numpy.dot(R, pr)
    ps = ps[0:3] / ps[3]
    ps += center
    return ps


def rotatePoints3D(pts, center, angle_x, angle_y, angle_z):
    """
    Transform points in 3D coordinates
    :param pts: point coordinates
    :param center: 3D center of rotation
    :param angle_x: angle around x-axis in deg
    :param angle_y: angle around y-axis in deg
    :param angle_z: angle around z-axis in deg
    :return: rotated points
    """
    ret = pts.copy()
    for i in xrange(pts.shape[0]):
        ret[i] = rotatePoint3D(pts[i], center, angle_x, angle_y, angle_z)
    return ret


def transformPoint3D(pt, M):
    """
    Transform point in 3D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt3 = numpy.dot(numpy.asarray(M).reshape((4, 4)), numpy.asarray([pt[0], pt[1], pt[2], 1]))
    return numpy.asarray([pt3[0] / pt3[3], pt3[1] / pt3[3], pt3[2] / pt3[3]])
