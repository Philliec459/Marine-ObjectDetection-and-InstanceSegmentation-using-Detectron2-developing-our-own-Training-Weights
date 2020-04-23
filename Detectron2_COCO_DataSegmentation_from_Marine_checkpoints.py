# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

 First, Import all variables
"""
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog

import matplotlib.pyplot as plt
import random
import os
import cv2


#from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

'''
 # download, decompress the data in Colab where much faster
 !wget https://cbpetro.s3.us-east-2.amazonaws.com/api/download/data.zip
 !unzip data.zip > /dev/null
'''

import wget

'''
  Download the data.zip file only once and then comment out the following 7 lines
'''
# url = 'https://cbpetro.s3.us-east-2.amazonaws.com/api/download/data.zip'
# wget.download(url, './data.zip')

# from zipfile import ZipFile
# zf = ZipFile('./data.zip', 'r')
# zf.extractall('./')
# zf.close()




'''
 Register coco instance
'''
from detectron2.data.datasets import register_coco_instances
register_coco_instances("nautical_ecp", {}, "./data/trainval.json", "./data/images")

nautical_metadata = MetadataCatalog.get("nautical_ecp")
dataset_dicts = DatasetCatalog.get("nautical_ecp")


'''
 Get ready to read models
'''
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("nautical_ecp",)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5


'''
 Load Weights
'''
cfg.OUTPUT_DIR = "./output"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("nautical_ecp", )
predictor = DefaultPredictor(cfg)



'''
 Random images from training set
'''
for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=nautical_metadata,
                   scale=0.8,
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels                                  
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()



'''
 Random images from boats set not used in training
'''
for i in range(2,29,1): 
    nautical_img_no = str(i)
    img = os.path.join("./data_val/images-" +nautical_img_no + ".jpeg") #1-24 in boats
    print(img)
    im  = cv2.imread(img)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=nautical_metadata, 
                   scale=0.8, 
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2_imshow(v.get_image()[:, :, ::-1])
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()





'''
 This is how you load and predict discrete images
'''
im = cv2.imread("./data_val/images-28.jpeg")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=nautical_metadata,
               scale=0.8,
               )
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()


