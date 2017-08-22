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
import cv2
from data.importers import DepthImporter, NYUImporter, ICVLImporter, MSRA15Importer
from util.helpers import rgb_to_gray

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
    Different evaluation metrics for hand pose, L2 distance used
    """

    def __init__(self, gtjoints, joints, dolegend=True, linewidth=1):
        """
        Initialize class

        :type gtjoints: groundtruth joints
        :type joints: calculated joints
        """

        if not (isinstance(gtjoints, numpy.ndarray) or isinstance(gtjoints, list)) or not (
                isinstance(joints, list) or isinstance(joints, numpy.ndarray)):
            raise ValueError("Params must be list or ndarray")

        if len(gtjoints) != len(joints):
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gtjoints), len(joints)))
            raise ValueError("Params must be the same size")

        if len(gtjoints) == len(joints) == 0:
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gtjoints), len(joints)))
            raise ValueError("Params must be of non-zero size")

        if gtjoints[0].shape != joints[0].shape:
            print("Error: groundtruth has {} dims, eval data has {}".format(gtjoints[0].shape, joints[0].shape))
            raise ValueError("Params must be of same dimensionality")

        self.gtjoints = numpy.asarray(gtjoints)
        self.joints = numpy.asarray(joints)
        assert (self.gtjoints.shape == self.joints.shape)

        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'brown', 'gray', 'indigo', 'pink',
                       'lightgreen', 'darkorange', 'peru', 'steelblue', 'turquoise']
        self.linestyles = ['-']  # , '--', '-.', ':', '-', '--', '-.', ':']
        self.linewidth = linewidth
        self.dolegend = dolegend
        self.default_plots = ['frameswithinmax', 'jointmeanerror', 'jointmaxerror']

        self.subfolder = './eval/'
        self.visiblemask = numpy.ones((self.gtjoints.shape[0], self.gtjoints.shape[1], 3))

        self.jointNames = None
        self.jointConnections = []
        self.jointConnectionColors = []
        self.plotMaxJointDist = 80
        self.plotMeanJointDist = 80
        self.plotMedianJointDist = 80
        self.VTKviewport = [0, 0, 0, 0, 0]

    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return numpy.nanmean(numpy.nanmean(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1))

    def getStdError(self):
        """
        get standard deviation of error over all joints, averaged over sequence
        :return: standard deviation of error
        """
        return numpy.nanmean(numpy.nanstd(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1))

    def getMeanErrorOverSeq(self):
        """
        get average error over all joints for each image of sequence
        :return: mean error
        """

        return numpy.nanmean(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1)

    def getMedianError(self):
        """
        get median error over all joints
        :return: median error
        """

        errs = numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2))
        return numpy.median(errs[numpy.isfinite(errs)])

    def getMaxError(self):
        """
        get max error over all joints
        :return: maximum error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)))

    def getMaxErrorOverSeq(self):
        """
        get max error over all joints for each image of sequence
        :return: maximum error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1)

    def getJointMeanError(self, jointID):
        """
        get error of one joint, averaged over sequence
        :param jointID: joint ID
        :return: mean joint error
        """

        return numpy.nanmean(numpy.sqrt(numpy.square(self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointStdError(self, jointID):
        """
        get standard deviation of one joint, averaged over sequence
        :param jointID: joint ID
        :return: standard deviation of joint error
        """

        return numpy.nanstd(numpy.sqrt(numpy.square(self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

    def getJointErrorOverSeq(self, jointID):
        """
        get error of one joint for each image of sequence
        :param jointID: joint ID
        :return: Euclidean joint error
        """

        return numpy.sqrt(numpy.square(self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1))

    def getJointDiffOverSeq(self, jointID):
        """
        get error of one joint for each image of sequence
        :param jointID: joint ID
        :return: joint error
        """

        return self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]

    def getJointMaxError(self, jointID):
        """
        get maximum error of one joint
        :param jointID: joint ID
        :return: maximum joint error
        """

        return numpy.nanmax(numpy.sqrt(numpy.square(self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))

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
        return (numpy.nanmax(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getNumFramesWithinMeanDist(self, dist):
        """
        calculate the number of frames where the mean difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmean(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getNumFramesWithinMedianDist(self, dist):
        """
        calculate the number of frames where the median difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.median(numpy.sqrt(numpy.square(self.gtjoints - self.joints).sum(axis=2)), axis=1) <= dist).sum()

    def getJointNumFramesWithinMaxDist(self, dist, jointID):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :param jointID: joint ID
        :return: number of frames
        """
        return (numpy.sqrt(numpy.square(self.gtjoints[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)) <= dist).sum()

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

        import matplotlib.pyplot as plt

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
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)  # ncol=2, prop={'size': 14})
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_frameswithin.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
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
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_joint_mean.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
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
        if self.dolegend:
            # Put a legend below current axis
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            bbea = (lgd,)
        else:
            bbea = None
        plt.show(block=False)
        fig.savefig('{}/{}_joint_max.pdf'.format(self.subfolder, basename), bbox_extra_artists=bbea,
                    bbox_inches='tight')
        plt.close(fig)

    def plotResult(self, dpt, gtcrop, joint, name=None, showGT=True, niceColors=False, showJoints=True, showDepth=True,
                   upsample=4., annoscale=1, block=False):
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

        # plot depth image with annotations
        if showDepth:
            imgcopy = dpt.copy()
            if len(imgcopy.shape) == 2:
                # display hack to hide nd depth
                msk = imgcopy > 0
                msk2 = imgcopy == 0
                min = imgcopy[msk].min()
                max = imgcopy[msk].max()
                imgcopy = (imgcopy - min) / (max - min) * 255.
                imgcopy[msk2] = 255.
        else:
            # same view as with image
            imgcopy = numpy.ones_like(dpt)*255.

        # resize image
        imgcopy = numpy.clip(imgcopy, 0., 255.)
        if len(imgcopy.shape) == 2:
            imgcopy = cv2.cvtColor(imgcopy.astype('uint8'), cv2.COLOR_GRAY2BGR)
        elif len(imgcopy.shape) == 3:
            imgcopy = imgcopy.astype('uint8')
        else:
            raise NotImplementedError("")

        if not numpy.allclose(upsample, 1):
            imgcopy = cv2.resize(imgcopy, dsize=None, fx=upsample, fy=upsample, interpolation=cv2.INTER_LINEAR)

        joint = joint.copy()
        for i in range(joint.shape[0]):
            joint[i, 0:2] -= numpy.asarray([dpt.shape[0]//2, dpt.shape[1]//2])
            joint[i, 0:2] *= upsample
            joint[i, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])

        gtcrop = gtcrop.copy()
        for i in range(gtcrop.shape[0]):
            gtcrop[i, 0:2] -= numpy.asarray([dpt.shape[0]//2, dpt.shape[1]//2])
            gtcrop[i, 0:2] *= upsample
            gtcrop[i, 0:2] += numpy.asarray([imgcopy.shape[0]//2, imgcopy.shape[1]//2])

        # use child class plots
        if showJoints:
            self.plotJoints(imgcopy, joint, annoscale=annoscale,
                            color=((0, 0, 255) if niceColors is False else 'nice'),
                            jcolor=((0, 0, 255) if niceColors is False else 'nice'))  # ours
        if showGT:
            if showJoints and showGT and (niceColors is True):
                cc = 'gray'
                jc = 'gray'
            elif niceColors is False:
                cc = (255, 0, 0)
                jc = (255, 0, 0)
            else:
                cc = 'nice'
                jc = 'nice'

            self.plotJoints(imgcopy, gtcrop, annoscale=annoscale,
                            color=cc, jcolor=jc)  # groundtruth
        
        if name is not None:
            cv2.imwrite('{}/annotated_{}.png'.format(self.subfolder, name), imgcopy)
        else:
            import matplotlib.pyplot as plt

            if plt.matplotlib.get_backend() == 'agg':
                return imgcopy
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.imshow(imgcopy)
                plt.tight_layout(pad=0)
                plt.show(block=block)
                return imgcopy

    def plotJoints(self, ax, joint, color='nice', jcolor=None, annoscale=1):
        """
        Plot connected joints
        :param ax: axis to plot on
        :param joint: joints to connect
        :param color: line color
        """

        if joint.shape[0] >= numpy.max(self.jointConnections):
            for i in range(len(self.jointConnections)):
                if isinstance(ax, numpy.ndarray):
                    if color == 'nice':
                        lc = tuple((self.jointConnectionColors[i]*255.).astype(int))
                    elif color == 'gray':
                        lc = tuple((rgb_to_gray(self.jointConnectionColors[i])*255.).astype(int))
                    else:
                        lc = color
                    cv2.line(ax, (int(numpy.rint(joint[self.jointConnections[i][0], 0])),
                                  int(numpy.rint(joint[self.jointConnections[i][0], 1]))),
                             (int(numpy.rint(joint[self.jointConnections[i][1], 0])),
                              int(numpy.rint(joint[self.jointConnections[i][1], 1]))),
                             lc, thickness=3*annoscale, lineType=cv2.CV_AA)
                else:
                    if color == 'nice':
                        lc = self.jointConnectionColors[i]
                    elif color == 'gray':
                        lc = rgb_to_gray(self.jointConnectionColors[i])
                    else:
                        lc = color
                    ax.plot(numpy.hstack((joint[self.jointConnections[i][0], 0], joint[self.jointConnections[i][1], 0])),
                            numpy.hstack((joint[self.jointConnections[i][0], 1], joint[self.jointConnections[i][1], 1])),
                            c=lc, linewidth=3.0*annoscale)
        for i in range(joint.shape[0]):
            if isinstance(ax, numpy.ndarray):
                if jcolor == 'nice':
                    jc = tuple((self.jointColors[i]*255.).astype(int))
                elif jcolor == 'gray':
                    jc = tuple((rgb_to_gray(self.jointColors[i])*255.).astype(int))
                else:
                    jc = jcolor
                cv2.circle(ax, (int(numpy.rint(joint[i, 0])), int(numpy.rint(joint[i, 1]))), 6*annoscale,
                           jc, thickness=-1, lineType=cv2.CV_AA)
            else:
                if jcolor == 'nice':
                    jc = self.jointColors[i]
                elif jcolor == 'gray':
                    jc = rgb_to_gray(self.jointColors[i])
                else:
                    jc = jcolor

                ax.scatter(joint[i, 0], joint[i, 1], marker='o', s=100,
                           c=jc)

    def plotResult3D(self, dpt, T, gt3Dorig, joint3D, filename=None, showGT=True, showPC=True, niceColors=False):
        """
        Plot 3D point cloud
        :param dpt: depth image
        :param T: 2D image transformation
        :param gt3Dorig: groundtruth 3D pose
        :param joint3D: 3D joint data
        :param filename: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param showPC: show point cloud
        :return: None, or image if filename=None
        """

        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        from util.vtkpointcloud import VtkPointCloud
        import matplotlib.pyplot as plt

        def close_window(iren):
            render_window = iren.GetRenderWindow()
            render_window.Finalize()
            iren.TerminateApp()

        def key_pressed_callback(obj, event):
            key = obj.GetKeySym()
            iren = obj
            render_window = iren.GetRenderWindow()
            if key == "s":
                file_name = self.subfolder + str(numpy.random.randint(0, 100)).zfill(5) + ".png"
                image = vtk.vtkWindowToImageFilter()
                image.SetInput(render_window)
                png_writer = vtk.vtkPNGWriter()
                png_writer.SetInputConnection(image.GetOutputPort())
                png_writer.SetFileName(file_name)
                render_window.Render()
                png_writer.Write()
            elif key == "c":
                camera = renderer.GetActiveCamera()
                print "Camera settings:"
                print "  * position:        %s" % (camera.GetPosition(),)
                print "  * focal point:     %s" % (camera.GetFocalPoint(),)
                print "  * up vector:       %s" % (camera.GetViewUp(),)

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

            pcl = self.getPCL(dpt, T)

            for k in xrange(pcl.shape[0]):
                point = pcl[k]
                pointCloud.addPoint(point)

            renderer.AddActor(pointCloud.vtkActor)
            renderer.ResetCamera()

        self.vtkPlotHand(renderer, joint3D, 'nice' if niceColors is True else (1, 0, 0))
        if showGT:
            self.vtkPlotHand(renderer, gt3Dorig, 'nice' if niceColors is True else (0, 0, 1))

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
        renderWindowInteractor.AddObserver("KeyPressEvent", key_pressed_callback)

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
        cb.actor = renderer.GetActors().GetLastActor()
        renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
        timerId = renderWindowInteractor.CreateRepeatingTimer(10)

        renderWindowInteractor.Start()

        if filename is not None:
            im = vtk.vtkWindowToImageFilter()
            writer = vtk.vtkPNGWriter()
            im.SetInput(renderWindow)
            im.Update()
            writer.SetInputConnection(im.GetOutputPort())
            writer.SetFileName('{}/{}.png'.format(self.subfolder, filename))
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

    def vtkPlotHand(self, renderer, joint3D, colors=(1, 0, 0)):
        """
        Plot hand in vtk renderer, as a stick and ball model
        :param renderer: vtk renderer instance
        :param joint3D: 3D joint locations
        :param colors: colors of joints or 'nice'
        :return: None
        """

        import vtk

        for i in range(joint3D.shape[0]):
            # create source
            source = vtk.vtkCubeSource()
            source.SetCenter(joint3D[i, 0], joint3D[i, 1], joint3D[i, 2])
            source.SetXLength(5.0)
            source.SetYLength(5.0)
            source.SetZLength(5.0)
            # mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color actor
            if colors == 'nice':
                actor.GetProperty().SetColor(self.jointColors[i][0],
                                             self.jointColors[i][1],
                                             self.jointColors[i][2])
            else:
                actor.GetProperty().SetColor(colors[0], colors[1], colors[2])

            # assign actor to the renderer
            renderer.AddActor(actor)

        if joint3D.shape[0] >= numpy.max(self.jointConnections):
            for i in range(len(self.jointConnections)):
                # create source
                source = vtk.vtkLineSource()
                source.SetPoint1(joint3D[self.jointConnections[i][0], 0], joint3D[self.jointConnections[i][0], 1], joint3D[self.jointConnections[i][0], 2])
                source.SetPoint2(joint3D[self.jointConnections[i][1], 0], joint3D[self.jointConnections[i][1], 1], joint3D[self.jointConnections[i][1], 2])

                # mapper
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(source.GetOutputPort())

                # actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                # color actor
                if colors == 'nice':
                    actor.GetProperty().SetColor(self.jointConnectionColors[i][0], self.jointConnectionColors[i][1], self.jointConnectionColors[i][2])
                else:
                    actor.GetProperty().SetColor(colors[0], colors[1], colors[2])

                actor.GetProperty().SetLineWidth(3)

                # assign actor to the renderer
                renderer.AddActor(actor)


class ICVLHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for ICVL dataset
    """

    def __init__(self, gt, joints, dolegend=True, linewidth=1):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(ICVLHandposeEvaluation, self).__init__(gt, joints, dolegend, linewidth)
        import matplotlib.colors

        # setup specific stuff
        self.jointNames = ['C', 'T1', 'T2', 'T3', 'I1', 'I2', 'I3', 'M1', 'M2', 'M3', 'R1', 'R2', 'R3', 'P1', 'P2', 'P3']
        self.jointColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 0, 0.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1.0]]]))[0, 0]]
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

    def getPCL(self, dpt, T):
        """
        Get pointcloud from frame
        :param dpt: depth image
        :param T: 2D transformation of crop
        """

        return ICVLImporter.depthToPCL(dpt, T)


class NYUHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for NYU dataset
    """

    def __init__(self, gt, joint, joints='eval', dolegend=True, linewidth=1):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(NYUHandposeEvaluation, self).__init__(gt, joint, dolegend, linewidth)
        import matplotlib.colors

        # setup specific stuff
        if joints == 'all':
            self.jointNames = ['P1', 'P2', 'P3', 'P4', 'P5', 'R1', 'R2', 'R3', 'R4', 'R5', 'M1', 'M2', 'M3', 'M4', 'M5',
                               'I1', 'I2', 'I3', 'I4', 'I5', 'T1', 'T2', 'T3', 'T4', 'T5', 'C1', 'C2', 'C3',
                               'W1', 'W2', 'W3', 'W4']
            self.jointColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.2]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.3]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.2]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.3]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.2]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.3]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.2]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.3]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.2]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.3]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0]]
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
            self.jointNames = ['P1', 'P2', 'R1', 'R2', 'M1', 'M2', 'I1', 'I2', 'T1', 'T2', 'T3', 'W1', 'W2', 'C']
            self.jointColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0],
                                matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 0, 0.0]]]))[0, 0]]
            self.jointConnections = [[13, 1], [1, 0], [13, 3], [3, 2], [13, 5], [5, 4], [13, 7], [7, 6], [13, 10],
                                     [10, 9], [9, 8], [13, 11], [13, 12]]
            self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1]]]))[0, 0]]
        else:
            raise ValueError("Unknown joint parameter")
        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 0, 180, 40]
        self.fps = 25.0

    def getPCL(self, dpt, T):
        """
        Get pointcloud from frame
        :param dpt: depth image
        :param T: 2D transformation of crop
        """

        return NYUImporter.depthToPCL(dpt, T)


class MSRAHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for ICVL dataset
    """

    def __init__(self, gt, joints, dolegend=True, linewidth=1):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(MSRAHandposeEvaluation, self).__init__(gt, joints, dolegend, linewidth)
        import matplotlib.colors

        # setup specific stuff
        self.jointNames = ['C', 'T1', 'T2', 'T3', 'T4', 'I1', 'I2', 'I3', 'I4', 'M1', 'M2', 'M3', 'M4', 'R1', 'R2',
                           'R3', 'R4', 'P1', 'P2', 'P3', 'P4']
        self.jointColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 0, 0.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1.0]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0],
                            matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1.0]]]))[0, 0]]
        self.jointConnections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                                 [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19],
                                 [19, 20]]
        self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0]]

        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 180, 40, 40]
        self.fps = 20.0

    def getPCL(self, dpt, T):
        """
        Get pointcloud from frame
        :param dpt: depth image
        :param T: 2D transformation of crop
        """

        return MSRA15Importer.depthToPCL(dpt, T)
