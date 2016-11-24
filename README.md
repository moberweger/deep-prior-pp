# DeepPrior - Accurate and Fast 3D Hand Pose Estimation

Author: Markus Oberweger <oberweger@icg.tugraz.at>

## Requirements:
  * OS
    * Ubuntu 14.04
    * CUDA 7
  * via Ubuntu package manager:
    * python2.7
    * python-matplotlib
    * python-scipy
    * python-pil
    * python-numpy
    * python-vtk6
    * python-pip
    * python-vtk6
  * via pip install:
    * scikit-learn
    * progressbar
    * psutil
    * theano (0.8)
  * Camera driver
    * OpenNI for Kinect
    * DepthSense SDK for Creative Senz3D. 

For a description of our method see:

M. Oberweger, P. Wohlhart, and V. Lepetit. Hands Deep in Deep Learning for Hand Pose Estimation. In Computer Vision Winter Workshop, 2015.

## Setup:
  * Put dataset files into ./data (e.g. [ICVL dataset](http://www.iis.ee.ic.ac.uk/~dtang/hand.html), or [NYU dataset](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm) )
  * Goto ./src and see the main file test_realtimepipeline.py how to handle the API
  * Camera interface for the Creative Senz3D is included in ./src/util. Build them with `cmake . && make`.

## Pretrained models:
[Download](https://webadmin.tugraz.at/fileadmin/user_upload/Institute/ICG/Downloads/team_lepetit/3d_hand_pose/DeepPrior_pretrained.zip) pretrained models for ICVL and NYU dataset.

## Datasets:
The ICVL dataset is trained for a time-of-flight camera, and the NYU dataset for a structured light camera. The annotations are different. See the papers for it.

D. Tang, H. J. Chang, A. Tejani, and T.-K. Kim. Latent Regression Forest: Structured Estimation of 3D Articulated Hand Posture. In Conference on Computer Vision and Pattern Recognition, 2014.

J. Tompson, M. Stein, Y. LeCun, and K. Perlin. Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks. ACM Transactions on Graphics, 33, 2014.
