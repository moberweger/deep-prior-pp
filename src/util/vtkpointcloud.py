"""Provides class for handling point clouds in VTK.

VtkPointCloud resembles a point cloud for display in VTK.
Use to manage the 3D points.

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

import vtk
import numpy

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class VtkPointCloud:
    """
    Manage 3D point cloud in VTK
    @see: http://sukhbinder.wordpress.com/2013/09/17/python-vtk-script-to-display-3d-xyz-data/
    """
    def __init__(self, pts=None, zMin=-10.0, zMax=10.0, maxNumPoints=1e6, color='depth'):
        """
        Initialize class
        :param zMin: minimum depth
        :param zMax: maximum depth
        :param maxNumPoints: maximum number of points
        :return: None
        """
        self.color = color
        self.maxNumPoints = int(maxNumPoints)
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)
        self.vtkActor.GetProperty().SetPointSize(3.0)
        self.rng = numpy.random.RandomState(23455)

        if pts is not None:
            self.addPoints(pts)
 
    def addPoint(self, point):
        """
        Add point to point cloud, if more than maximum points are set, they are randomly subsampled
        :param point: 3D coordinates
        :return: None
        """
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            if self.color == 'depth':
                self.vtkDepth.InsertNextValue(point[2])
            else:
                import numbers
                assert isinstance(self.color, numbers.Number)
                self.vtkDepth.InsertNextValue(self.color)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = self.rng.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def addPoints(self, points):
        """
        Add points to the point cloud
        :param points: Nx3 matrix with points
        :return: None
        """
        assert len(points.shape) == 2, points.shape
        assert points.shape[1] == 3, points.shape
        for k in xrange(points.shape[0]):
            self.addPoint(points[k])
 
    def clearPoints(self):
        """
        Clear all points from the point cloud
        :return: None
        """
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

    @staticmethod
    def viewer(pointclouds):
        assert all([isinstance(p, VtkPointCloud) for p in pointclouds])

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)

        for p in pointclouds:
            renderer.AddActor(p.vtkActor)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Begin Interaction
        renderWindow.Render()
        renderWindow.SetWindowName("XYZ Data Viewer")

        renderWindowInteractor.Start()
