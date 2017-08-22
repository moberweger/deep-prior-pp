# DeepPrior++: Improving Fast and Accurate 3D Hand Pose Estimation

Author: Markus Oberweger <oberweger@icg.tugraz.at>

## Requirements:
  * OS
    * Ubuntu 14.04
    * CUDA 7 + cuDNN 5
  * via Ubuntu package manager:
    * python2.7
    * python-matplotlib
    * python-scipy
    * python-pil
    * python-numpy
    * python-vtk6
    * python-pip
  * via pip install:
    * scikit-learn
    * progressbar
    * psutil
    * theano 0.9
  * Camera driver
    * OpenNI for Kinect
    * DepthSense SDK for Creative Senz3D

For a description of our method see:

M. Oberweger and V. Lepetit. DeepPrior++: Improving Fast and Accurate 3D Hand Pose Estimation. In ICCV Workshop, 2017.

## Setup:
  * Put dataset files into ./data (e.g. [ICVL](http://www.iis.ee.ic.ac.uk/~dtang/hand.html), or [MSRA](https://www.dropbox.com/s/bmx2w0zbnyghtp7/cvpr15_MSRAHandGestureDB.zip?dl=0) (thanks to @geliuhao for providing), or [NYU](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) dataset)
  * Goto ./src and see the main file test_realtimepipeline.py how to handle the API
  * Camera interface for the Creative Senz3D is included in ./src/util. Build them with `cmake . && make`.

## Pretrained models:
[Download](https://webadmin.tugraz.at/fileadmin/user_upload/Institute/ICG/Downloads/team_lepetit/3d_hand_pose/DeepPriorPP_pretrained.zip) pretrained models for ICVL and NYU dataset.

## Datasets:
The ICVL and MSRA dataset is trained for a time-of-flight camera, and the NYU dataset for a structured light camera. The annotations are different. See the papers for it.

D. Tang, H. J. Chang, A. Tejani, and T.-K. Kim. Latent Regression Forest: Structured Estimation of 3D Articulated Hand Posture. In Conference on Computer Vision and Pattern Recognition, 2014.

X. Sun, Y. Wei, S. Liang, X. Tang and J. Sun. Cascaded Hand Pose Regression. In Conference on Computer Vision and Pattern Recognition, 2015.

J. Tompson, M. Stein, Y. LeCun, and K. Perlin. Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks. ACM Transactions on Graphics, 33, 2014.
