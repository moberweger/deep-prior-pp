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


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = numpy.prod([x.size for x in arrays])
    if out is None:
        out = numpy.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = numpy.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out


def shuffle_many_inplace(arrays, random_state=None):
    """
    Shuffle given list of array consistently along first dimension
    :param arrays: list of arrays
    :param random_state: random state or seed
    :return: None
    """

    if random_state is None:
        rng = numpy.random.mtrand._rand
    elif isinstance(random_state, numpy.random.RandomState):
        rng = random_state
    else:
        raise ValueError("random_state must be None or numpy RandomState")

    assert all(i.shape[0] == arrays[0].shape[0] for i in arrays[1:])

    # Fisher-Yates Shuffle
    for oi in reversed(range(1, arrays[0].shape[0])):
        ni = rng.randint(oi+1)
        for a in arrays:
            a[[oi, ni]] = a[[ni, oi]]


def gaussian_kernel(kernel_shape, sigma=None):
    """
    Get 2D Gaussian kernel
    :param kernel_shape: kernel size
    :param sigma: sigma of Gaussian distribution
    :return: 2D Gaussian kernel
    """
    kern = numpy.zeros((kernel_shape, kernel_shape), dtype='float32')

    # get sigma from kernel size
    if sigma is None:
        sigma = 0.3*((kernel_shape-1.)*0.5 - 1.) + 0.8

    def gauss(x, y, s):
        Z = 2. * numpy.pi * s ** 2.
        return 1. / Z * numpy.exp(-(x ** 2. + y ** 2.) / (2. * s ** 2.))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            kern[i, j] = gauss(i - mid, j - mid, sigma)

    return kern / numpy.sum(kern)


def rgb_to_gray(rgb):
    """
    Convert rgb color to gray
    """
    assert len(rgb) == 3, "rgb should be 3, got {}".format(len(rgb))
    g = 0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2]
    return numpy.asarray([g, g, g])


def chunks(l, n):
    """
    Yield successive n-sized chunks from l
    :param l: list
    :param n: size of chunk
    :return: iterator
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]
