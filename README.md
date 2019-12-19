# PersonMatching
Person matching in photo collections. 

## Objective 
The objective of this project is to retrieve the photos that belong to a certain individual given a query image. <br>
* Person matching is a problem of matching feature descriptor that describes a person across images consisting of multiple objects(people). <br>
* This problem typically arises in face detection and matching computer vision tasks. It’s applications include security surveillance, Google Photos,  apps where they find similar faces etc.<br>
* We pose the problem of person matching as an image retrieval task with predefined probe and gallery set.<br>

This repo is forked from the facenet-pytorch(timesler/facenet-pytorch) as we are using the already developed ResNet model to get the feature descriptors. Our code is under the PersonMatching folder.

## Download the Dataset
* Since the PIPA dataset is large in size, we haven't added it to this repo. Please download the tar files from [here](https://people.eecs.berkeley.edu/~nzhang/piper.html) into a data folder in the Person Matching Code directory. <br>

* Also download the [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) dataset into the same data folder in the Person Matching Code directory. <br>

## Unzipping tar files for the data
tar -xvf data/pipa_test.tar <br>
tar -xvf data/pipa_val.tar <br>
unzip data/annotations.zip <br>
tar -xvf data/pipa_leftover.tar <br>

or run the shell script as: sh unzip_script.sh <br>

