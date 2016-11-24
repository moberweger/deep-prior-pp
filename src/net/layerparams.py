"""Provides LayerParams class used for parametrizing other layers.

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

import inspect

__author__ = "Paul Wohlhart <wohlhart@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Paul Wohlhart", "Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class LayerParams(object):
    """
    Parametrization of different layers for CNNs
    """

    def __init__(self, inputDim, outputDim):
        """
        Constructor
        """        
        self._inputDim = inputDim
        self._outputDim = outputDim
        
    @property
    def outputDim(self):
        return self._outputDim 

    @outputDim.setter
    def outputDim(self, value):
        self._outputDim = value
        self.update()

    @property
    def inputDim(self):
        return self._inputDim 

    @inputDim.setter
    def inputDim(self, value):
        self._inputDim = value
        self.update()
        
    def update(self):
        """
        Default. Override in derived
        """
        pass

    @property
    def activation_str(self):
        """
        Get printable string from activation function.
        :return: string
        """
        if hasattr(self, 'activation'):
            if self.activation is None:
                return str(None)
            elif inspect.isclass(self.activation):
                return self.activation.__class__.__name__
            elif inspect.isfunction(self.activation):
                return self.activation.__name__
            else:
                return str(self.activation)
        else:
            return ''
