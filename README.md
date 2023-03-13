# Deep Granulometry
## Image based estimation of concrete aggregate size distributions using deep learning
This repository contains the source code for the CNN based approach proposed in

_Coenen, M., Beyer, D., Ponick, A., Heipke, C., Haist, M., (2023): Deep Granulometry: Image based estimation of concrete aggregate size distributions using deep learning. To be published._

For details, we refer the reader to the mentioned paper.

[![Watch the video](https://github.com/MaximilianCoenen/MaxCoenen.github.io/blob/master/resources/images/klicktoplay2.png?raw=true)](https://maximiliancoenen.github.io/MaxCoenen.github.io/resources/images/2023_DeepGranulometry.mp4)


## Info
The code contains the architecture of the network and an example for the training routine (cf. main_Train.py). It is built using keras and tensorflow backend.

It can be used for both, grading curve classification and grading curve regression. Please adapt the settings and variables in main_Train.py accordingly.

### Classification
For classification purposes, set the task variable in main_Train.py to 'Classification'. Training requires training data in form of images and a grading curve class associated to each image. The images have to be located in one folder with 'fldrImgTrain' linking to the path of that folder and a folder 'fldrLabel' which to contain .txt files with the same name as the images. Each txt file contains a single integer value between [0, nClasses] corresponding to the class of the image.

An example data set for classification purposes of concrete aggregate can be found here:  https://doi.org/10.25835/etbkk0pb

### Regression
For regression purposes (prediction of size distribution percentiles), the task variable in main_Train.py has to be set to 'Regression'. To run training, a training data set is required. To this end, fldrImgTrain has to be set to a folder path containing the training images and fldrLabel has to lead to a path containing a .txt file for each image which comprises the percentile values of the size distributions.

An example data set for regression purposes of concrete aggregate can be found here: https://doi.org/10.25835/61y9peiq

### Related publications
If you make use of the code and/or data, please cite the publications listed below. 

* **Coenen, M., Beyer, D., Ponick, A., Heipke, C. and Haist, M., 2023**: Deep Granulometry: Image based estimation of concrete aggregate size distributions using deep learning. _To be published_.

* **Coenen, M., Beyer, D., Heipke, C. and Haist, M., 2022**: Learning to Sieve: Prediction of Grading Curves from Images of Concrete Aggregate. In: _ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences V-2-2022_, pp. 227-235, [Link](https://doi.org/10.5194/isprs-annals-V-2-2022-227-2022).
