"""Provides class for evaluating hand pose accuracy.

HandposeEvaluation provides interface for evaluating the hand pose accuracy.
ICVLHandposeEvaluation, NYUHandposeEvaluation are specific instances for different datasets.

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
import matplotlib.pyplot as plt
from data.basetypes import ICVLFrame
from data.importers import DepthImporter, NYUImporter, ICVLImporter
from mpl_toolkits.mplot3d import Axes3D
from data.transformations import transformPoint2D
import progressbar as pb
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from util.vtkpointcloud import VtkPointCloud
from sklearn import mixture

__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"


class HandposeEvaluation(object):
    """
    Different evaluation metrics for handpose, L2 distance used
    """

    def __init__(self, gt, joints):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        if not (isinstance(gt, numpy.ndarray) or isinstance(gt, list)) or not (
                isinstance(joints, list) or isinstance(joints, numpy.ndarray)):
            raise ValueError("Params must be list or ndarray")

        if len(gt) != len(joints):
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be the same size")

        if len(gt) == len(joints) == 0:
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be of non-zero size")

        if gt[0].shape != joints[0].shape:
            print("Error: groundtruth has {} dims, eval data has {}".format(gt[0].shape, joints[0].shape))
            raise ValueError("Params must be of same dimensionality")

        self.gt = numpy.asarray(gt)
        self.joints = numpy.asarray(joints)
        assert (self.gt.shape == self.joints.shape)

        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'brown', 'gray', 'indigo', 'pink',
                       'lightgreen', 'darkorange', 'peru', 'steelblue', 'turquoise']
        self.linestyles = ['-']  # , '--', '-.', ':', '-', '--', '-.', ':']
        self.jointcolors = [(0.0, 0.0, 1.0), (0.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.75, 0.75),
                            (0.75, 0, 0.75), (0.75, 0.75, 0), (0.0, 0.0, 0.0)]

        self.subfolder = './eval/'
        self.visiblemask = numpy.ones((self.gt.shape[0], self.gt.shape[1], 3))

        self.jointNames = None
        self.jointConnections = []
        self.jointConnectionColors = []
        self.plotMaxJointDist = 80
        self.plotMeanJointDist = 80
        self.plotMedianJointDist = 80
        self.VTKviewport = [0, 0, 0, 0, 0]

    def getJointNumFramesVisible(self, jointID):
        """
        Get number of frames in which joint is visible
        :param jointID: joint ID
        :return: number of frames
        """

        return numpy.nansum(self.gt[:, jointID, :]) / self.gt.shape[2]  # 3D

    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return numpy.nanmean(numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1))

    def getStdError(self):
        """
        get standard deviation of error over all joints, averaged over sequence
        :return: standard deviation of error
        """
        return numpy.nanmean(numpy.nanstd(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1))

    def getMeanErrorOverSeq(self):
        """
        get average error over all joints for each image of sequence
        :return: mean error
        """

        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1)

    def getMaxError(self):
        """
        get max error over all joints
        :return: maximum error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)))

    def getMaxErrorOverSeq(self):
        """
        get max error over all joints for each image of sequence
        :return: maximum error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1)

    def getJointMeanError(self, jointID):
        """
        get error of one joint, averaged over sequence
        :param jointID: joint ID
        :return: mean joint error
        """

        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointStdError(self, jointID):
        """
        get standard deviation of one joint, averaged over sequence
        :param jointID: joint ID
        :return: standard deviation of joint error
        """

        return numpy.nanstd(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointErrorOverSeq(self, jointID):
        """
        get error of one joint for each image of sequence
        :param jointID: joint ID
        :return: Euclidean joint error
        """

        return numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1))

    def getJointMaxError(self, jointID):
        """
        get maximum error of one joint
        :param jointID: joint ID
        :return: maximum joint error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def cumulativeMovingAverage(self, data):
        """
        calculate cumulative moving average from data
        :param data: 1D input data
        :return: cumulative average
        """

        new_avg = numpy.zeros((data.shape[0], 1), float)
        new_avg[0] = data[0]
        for i in range(1, data.shape[0]):
            new_avg[i] = numpy.nanmean(data[0:i])

        return new_avg

    def getNumFramesWithinMaxDist(self, dist):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getNumFramesWithinMeanDist(self, dist):
        """
        calculate the number of frames where the mean difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getNumFramesWithinMedianDist(self, dist):
        """
        calculate the number of frames where the median difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.median(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getJointNumFramesWithinMaxDist(self, dist, jointID):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :param jointID: joint ID
        :return: number of frames
        """
        return (numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)) <= dist).sum()

    def getMDscore(self, dist):
        """
        calculate the max dist score, ie. MD=\int_0^d{\frac{|F<x|}{|F|}dx = \sum
        :param dist: distance between joint and GT
        :return: score value [0-1]
        """
        vals = [(numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= j).sum() / float(self.joints.shape[0]) for j in range(0, dist)]
        return numpy.asarray(vals).sum() / float(dist)

    def plotEvaluation(self, basename, methodName='Our method', baseline=None):
        """
        plot and save standard evaluation plots
        :param basename: file basename
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :return: None
        """

        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(self.__class__.__name__,
                                                                                         bs[1].__class__.__name__))

        # plot number of frames within max distance
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMaxJointDist)],
                label=methodName, c=self.colors[0], linestyle=self.linestyles[0])
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMaxJointDist)],
                        label=bs[0], c=self.colors[bs_idx % len(self.colors)], linestyle=self.linestyles[bs_idx % len(self.linestyles)])
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of frames within distance / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig('{}/{}_frameswithin.pdf'.format(self.subfolder,basename), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)

        # plot mean error for each joint
        ind = numpy.arange(self.joints.shape[1]+1)  # the x locations for the groups, +1 for mean
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        mean = [self.getJointMeanError(j) for j in range(self.joints.shape[1])]
        mean.append(self.getMeanError())
        std = [self.getJointStdError(j) for j in range(self.joints.shape[1])]
        std.append(self.getStdError())
        ax.bar(ind, numpy.array(mean), width, label=methodName, color=self.colors[0])  # , yerr=std)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                mean = [bs[1].getJointMeanError(j) for j in range(self.joints.shape[1])]
                mean.append(bs[1].getMeanError())
                std = [bs[1].getJointStdError(j) for j in range(self.joints.shape[1])]
                std.append(bs[1].getStdError())
                ax.bar(ind + width * float(bs_idx), numpy.array(mean), width,
                       label=bs[0], color=self.colors[bs_idx % len(self.colors)])  # , yerr=std)
                bs_idx += 1
        ax.set_xticks(ind + width)
        ll = list(self.jointNames)
        ll.append('Avg')
        label = tuple(ll)
        ax.set_xticklabels(label)
        plt.ylabel('Mean error of joint / mm')
        # plt.ylim([0.0,50.0])
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig('{}/{}_joint_mean.pdf'.format(self.subfolder, basename), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)

        # plot maximum error for each joint
        ind = numpy.arange(self.joints.shape[1])  # the x locations for the groups
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        ax.bar(ind, numpy.array([self.getJointMaxError(j) for j in range(self.joints.shape[1])]), width,
               label=methodName, color=self.colors[0])
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.bar(ind + width * float(bs_idx),
                       numpy.array([bs[1].getJointMaxError(j) for j in range(self.joints.shape[1])]), width,
                       label=bs[0], color=self.colors[bs_idx % len(self.colors)])
                bs_idx += 1
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.jointNames)
        plt.ylabel('Maximum error of joint / mm')
        plt.ylim([0.0, 200.0])
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig('{}/{}_joint_max.pdf'.format(self.subfolder, basename), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)

    def plotResult(self, dpt, gtcrop, joint, name=None, showGT=True, niceColors=False, showJoints=True, showDepth=True):
        """
        Show the annotated depth image
        :param dpt: depth image to show
        :param gtcrop: cropped 2D coordinates
        :param joint: joint data
        :param name: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param niceColors: plot nice gradient colors for each joint
        :return: None, or image if name = None
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        # plot depth image with annotations
        if showDepth:
            imgcopy = dpt.copy()
            # display hack to hide nd depth
            msk = imgcopy > 0
            msk2 = imgcopy == 0
            min = imgcopy[msk].min()
            max = imgcopy[msk].max()
            imgcopy = (imgcopy - min) / (max - min) * 255.
            imgcopy[msk2] = 255.
            ax.imshow(imgcopy, cmap='gray')
        else:
            # same view as with image
            ax.set_xlim([0, dpt.shape[0]])
            ax.set_ylim([0, dpt.shape[1]])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()
        # use child class plots
        if showJoints:
            self.plotJoints(ax, joint, color=('r' if niceColors is False else 'nice'), jcolor=('r' if niceColors is True else None))  # ours
        if showGT:
            self.plotJoints(ax, gtcrop, color=('b' if niceColors is False else 'nice'), jcolor=('b' if niceColors is True else None))  # groundtruth
        plt.tight_layout(pad=0)
        plt.show(block=False)
        if name is not None:
            fig.savefig('{}/annotated_{}.png'.format(self.subfolder, name), bbox_inches='tight')
            plt.close(fig)
        else:
            # If we haven't already shown or saved the plot, then we need to draw the figure first...
            fig.patch.set_facecolor('w')
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data

    def plotJoints(self, ax, joint, color='nice', jcolor=None):
        """
        Plot connected joints
        :type ax: axis to plot on
        :type joint: joints to connect
        :type color: line color
        """

        color_index = 0
        for i in range(joint.shape[0]):
            ax.scatter(joint[i, 0], joint[i, 1], c=(self.jointcolors[color_index % len(self.jointcolors)] if jcolor is None else jcolor), marker='.', s=400)
            color_index += 1
        for i in range(len(self.jointConnections)):
            ax.plot(numpy.hstack((joint[self.jointConnections[i][0], 0], joint[self.jointConnections[i][1], 0])),
                    numpy.hstack((joint[self.jointConnections[i][0], 1], joint[self.jointConnections[i][1], 1])),
                    c=(color if color is not 'nice' else self.jointConnectionColors[i]), linewidth=2.0)

    def plotResult3D(self, frame, joint3D, filename=None, showGT=True, showPC=True, niceColors=False):
        """
        Plot 3D point cloud
        :param frame: icvlframe to show
        :param joint3D: 3D joint data
        :param filename: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param showPC: show point cloud
        :return: None, or image if filename=None
        """
        class vtkTimerCallback():
            def __init__(self):
                pass

            def execute(self, obj, event):
                if plt.matplotlib.get_backend() == 'agg':
                    iren = obj
                    render_window = iren.GetRenderWindow()
                    render_window.Finalize()
                    iren.TerminateApp()
                    del render_window, iren

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)

        if showPC is True:
            pointCloud = VtkPointCloud()

            pcl = self.getPCL(frame)

            for k in xrange(pcl.shape[0]):
                point = pcl[k]
                pointCloud.addPoint(point)

            renderer.AddActor(pointCloud.vtkActor)
            renderer.ResetCamera()

        # setup camera position
        camera = renderer.GetActiveCamera()
        camera.Pitch(self.VTKviewport[0])
        camera.Yaw(self.VTKviewport[1])
        camera.Roll(self.VTKviewport[2])
        camera.Azimuth(self.VTKviewport[3])
        camera.Elevation(self.VTKviewport[4])

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        for i in range(joint3D.shape[0]):
            # create source
            source = vtk.vtkSphereSource()
            source.SetCenter(joint3D[i, 0], joint3D[i, 1], joint3D[i, 2])
            source.SetRadius(5.0)

            # mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInput(source.GetOutput())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color actor
            if not niceColors:
                actor.GetProperty().SetColor(1, 0, 0)
            else:
                actor.GetProperty().SetColor(self.jointcolors[i % len(self.jointcolors)][0],
                                             self.jointcolors[i % len(self.jointcolors)][1],
                                             self.jointcolors[i % len(self.jointcolors)][2])

            # assign actor to the renderer
            renderer.AddActor(actor)

            if showGT:
                # create source
                source = vtk.vtkSphereSource()
                source.SetCenter(frame.gt3Dorig[i, 0], frame.gt3Dorig[i, 1], frame.gt3Dorig[i, 2])
                source.SetRadius(5.0)

                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInput(source.GetOutput())

                # actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # color actor
                if not niceColors:
                    actor.GetProperty().SetColor(0, 0, 1)
                else:
                    actor.GetProperty().SetColor(self.jointcolors[i % len(self.jointcolors)][0],
                                                 self.jointcolors[i % len(self.jointcolors)][1],
                                                 self.jointcolors[i % len(self.jointcolors)][2])

                # assign actor to the renderer
                renderer.AddActor(actor)

        for i in range(len(self.jointConnections)):
            # create source
            source = vtk.vtkLineSource()
            source.SetPoint1(joint3D[self.jointConnections[i][0], 0], joint3D[self.jointConnections[i][0], 1], joint3D[self.jointConnections[i][0], 2])
            source.SetPoint2(joint3D[self.jointConnections[i][1], 0], joint3D[self.jointConnections[i][1], 1], joint3D[self.jointConnections[i][1], 2])

            # mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInput(source.GetOutput())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color actor
            if not niceColors:
                actor.GetProperty().SetColor(1, 0, 0)
            else:
                actor.GetProperty().SetColor(self.jointConnectionColors[i][0], self.jointConnectionColors[i][1], self.jointConnectionColors[i][2])
            actor.GetProperty().SetLineWidth(3)

            # assign actor to the renderer
            renderer.AddActor(actor)

            if showGT:
                # create source
                source = vtk.vtkLineSource()
                source.SetPoint1(frame.gt3Dorig[self.jointConnections[i][0], 0], frame.gt3Dorig[self.jointConnections[i][0], 1], frame.gt3Dorig[self.jointConnections[i][0], 2])
                source.SetPoint2(frame.gt3Dorig[self.jointConnections[i][1], 0], frame.gt3Dorig[self.jointConnections[i][1], 1], frame.gt3Dorig[self.jointConnections[i][1], 2])

                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInput(source.GetOutput())

                # actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # color actor
                if not niceColors:
                    actor.GetProperty().SetColor(0, 0, 1)
                else:
                    actor.GetProperty().SetColor(self.jointConnectionColors[i][0], self.jointConnectionColors[i][1], self.jointConnectionColors[i][2])
                actor.GetProperty().SetLineWidth(3)

                # assign actor to the renderer
                renderer.AddActor(actor)

        if showPC is False:
            renderer.ResetCamera()
            # setup camera position
            camera = renderer.GetActiveCamera()
            camera.Pitch(self.VTKviewport[0])
            camera.Yaw(self.VTKviewport[1])
            camera.Roll(self.VTKviewport[2])
            camera.Azimuth(self.VTKviewport[3])
            camera.Elevation(self.VTKviewport[4])

        # Begin Interaction
        renderWindow.Render()
        renderWindow.SetWindowName("XYZ Data Viewer")

        # Sign up to receive TimerEvent
        cb = vtkTimerCallback()
        cb.actor = actor
        renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(10)

        renderWindowInteractor.Start()

        if filename is not None:
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(renderWindow)
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName('{}/pointcloud_{}.png'.format(self.subfolder, filename))
            writer.Write()
        else:
            im = vtk.vtkWindowToImageFilter()
            im.SetInput(renderWindow)
            im.Update()
            vtk_image = im.GetOutput()
            height, width, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            return vtk_to_numpy(vtk_array).reshape(height, width, components)


class ICVLHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for ICVL dataset
    """

    def __init__(self, gt, joints):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(ICVLHandposeEvaluation, self).__init__(gt, joints)
        import matplotlib

        # setup specific stuff
        self.jointNames = ('C', 'T1', 'T2', 'T3', 'I1', 'I2', 'I3', 'M1', 'M2', 'M3', 'R1', 'R2', 'R3', 'P1', 'P2', 'P3')
        self.jointConnections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [0, 10],
                                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]
        self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0]]

        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 180, 40, 40]
        self.fps = 10.0

    def getPCL(self, frame):
        """
        Get pointcloud from frame

        :type frame: image frame
        """

        return ICVLImporter.depthToPCL(frame.dpt, frame.T)


class NYUHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for NYU dataset
    """

    def __init__(self, gt, joint, joints='eval'):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(NYUHandposeEvaluation, self).__init__(gt, joint)
        import matplotlib

        # setup specific stuff
        if joints == 'all':
            self.jointNames = ('P1', 'P2', 'P3', 'P4', 'P5', 'R1', 'R2', 'R3', 'R4', 'R5', 'M1', 'M2', 'M3', 'M4', 'M5',
                               'I1', 'I2', 'I3', 'I4', 'I5', 'T1', 'T2', 'T3', 'T4', 'T5', 'C1', 'C2', 'C3',
                               'W1', 'W2', 'W3', 'W4')
            self.jointConnections = [[33, 5], [5, 4], [4, 3], [3, 2], [2, 1], [1, 0],
                                     [32, 11], [11, 10], [10, 9], [9, 8], [8, 7], [7, 6],
                                     [32, 17], [17, 16], [16, 15], [15, 14], [14, 13], [13, 12],
                                     [32, 23], [23, 22], [22, 21], [21, 20], [20, 19], [19, 18],
                                     [34, 29], [29, 28], [28, 27], [27, 26], [26, 25], [25, 24],
                                     [34, 32], [34, 33], [33, 32],
                                     [34, 30], [34, 31], [35, 30], [35, 31]]
            self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0]]

        elif joints == 'eval':
            self.jointNames = ('P1', 'P2', 'R1', 'R2', 'M1', 'M2', 'I1', 'I2', 'T1', 'T2', 'T3', 'W1', 'W2', 'C')
            self.jointConnections = [[13, 1], [1, 0], [13, 3], [3, 2], [13, 5], [5, 4], [13, 7], [7, 6], [13, 10],
                                     [10, 9], [9, 8], [13, 11], [13, 12]]
            self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1]]]))[0, 0]]

        elif joints == 'real':  # TODO
            raise NotImplementedError("!")
            self.jointNames = ()
            self.jointConnections = []
            self.jointConnectionColors = []
        else:
            raise ValueError("Unknown joint parameter")
        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 0, 180, 40]
        self.fps = 25.0

    def getPCL(self, frame):
        """
        Get pointcloud from frame

        :type frame: image frame
        """

        return NYUImporter.depthToPCL(frame.dpt, frame.T)

