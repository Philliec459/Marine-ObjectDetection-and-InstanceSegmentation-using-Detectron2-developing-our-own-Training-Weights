# Marine Image Segmentation using Detectron2
This repository was inspired by the following Detectron2 tutorial:

#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

and this tutorial is what led to this repository.

According to Facebook AI Research (FAIR), "Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms". 

https://github.com/facebookresearch/detectron2

## Objectives
The objective of this project is to recognize nautical objects like buoys, ships or land while at sea using a video camera for real-time warnings. We plan on installing a low power drain Jetson Xavier on a boat with a video camera to detect these nautical objects. This project is actually a follow up to Adrian Llopart Maurin's GitHub repository where he used Mask RCNN to train and predict Marine objects typically encountered at sea. Adrian labeled his images (with labelme?) to detect ships, buoys, land, sea and sky. Other ships, buoys and land are all important to avoid these objects while boating. 

https://github.com/Allopart/Maritme_Mask_RCNN

We used Adrian's Marine image dataset as our training set, which also included all of his the labelme .json file labels.  Adrian used a Panoptic Segmentation approach were he labeled the entire image for ships, buoys, land, sea and sky. For this repository; however, we edited Adrian's nautical images and labels in labelme and eliminated sea and sky. For our next panoptic segmentation project, we will go back to Arian's comprehensive labeling approach.


## Marine Labels
The following image is an example of an image used as a training images. This image was captured while we were using labelme.

![Image](labelme.png)

Adrian's training dataset were images and separate .json for each image. We used Chengwei Zhang's GitHub repository (see link below) to combine all the individual .json files for each image. Labelme creates individual json files for each image. With Chengwei's program we created the single trainval.json file used in training by Detectron2. We ran this program using the following command line: 

  python labelme2coco.py .data/images

https://github.com/Tony607/labelme2coco

Chengwei also included a notebook that was used to evaluate and scan the individual image results as a qc check before starting the training. This notebook is called COCO_Image_Viewer.ipynb, and it too is included in this repository.

## Data used for Training
A subset to Adrian's images are accessed from the following AWS source:

https://cbpetro.s3.us-east-2.amazonaws.com/api/download/data.zip

This creates a folder called data and within this folder is the single .json file and a folder called "images". 

We used the following Colab notebook to train this nautical dataset on Colab:

Detectron2_custom_coco_data_instance_segmentation_marine_ver5_TRAIN_Colab.ipynb

Colab generates the new weights that are are stored in the Colab "/output/model_final.pth" directory. After training this file can be accessed in Colab by tapping on the folder icon to the very far left of the Table of Contents in Colab. The weights can be downloaded to your Download directory and placed in your project directory as "output/model_fina.pth". 

Our training weights can be downloaded from the following AWS link. Just click on it. 

https://cbpetro.s3.us-east-2.amazonaws.com/api/download/output.zip

## Predict Nautical Objects
We used the following notebook to then predict our nautical objects like buoys, ships and land features:

Detectron2_COCO_DataSegmentation_from_Marine_checkpoints.ipynb

or you can use the following python code in Spyder:

Detectron2_COCO_DataSegmentation_from_Marine_checkpoints.py

![Image](results.png)

We have also provided a data_val subdirectory in this GitHub repository. This folder has numerous images of nautical objects, all of which were not used in training. These images are being used as validation of our model. The instance segmentation process appears to be working rather well as can be observed in the above image. 

![Image](composite.png)

At this time we are a bit unclear as to how to use a validation set in the Detectron2 training process. This will be added at a later date.  


