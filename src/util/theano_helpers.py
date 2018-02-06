"""
This is the file for diverse helper functions.

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

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


EPS = numpy.cast[numpy.float32](3.*numpy.finfo(numpy.float32).eps)
PI = numpy.cast[numpy.float32](numpy.pi)


def sigmoid(x):
    """
    Sigmoid unit
    :param x: input value
    :return: sigmoid(x)
    """
    import theano.tensor as T

    return T.nnet.sigmoid(x)


def tanh(x):
    """
    Tanh unit
    :param x: input value
    :return: tanh(x)
    """
    import theano.tensor as T

    return T.tanh(x)


def ReLU(x):
    """
    Rectified linear unit
    :param x: input value
    :return: max(x, 0)
    """
    import theano.tensor as T

    return T.maximum(x, 0)  # this version is slightly slower, but has defined gradients, not as theano version
    # return T.nnet.relu(x, 0)
