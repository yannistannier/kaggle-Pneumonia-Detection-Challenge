# coding: utf-8

# **Mask-RCNN Sample Starter Model for the RSNA Pneumonia Detection Challenge**
#
# This notebook is developed by [MD.ai](https://www.md.ai), the platform for medical AI.  MD.ai is invovled in developing this dataset with RSNA.
# This notebook covers the basics of parsing the competition dataset, training using a detector basd on the [Mask-RCNN algorithm](https://arxiv.org/abs/1703.06870) for object detection and instance segmentation.
# **Note that the Mask-RCNN detector configuration parameters have been selected to reduce training time for demonstration purposes, they are not optimal.**
#
# This is based on our deep learning for medical imaging lessons:
#
# - Lesson 1. Classification of chest vs. adominal X-rays using TensorFlow/Keras [Github](https://github.com/mdai/ml-lessons/blob/master/lesson1-xray-images-classification.ipynb) [Annotator](https://public.md.ai/annotator/project/PVq9raBJ)
# - Lesson 2. Lung X-Rays Semantic Segmentation using UNets. [Github](https://github.com/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb)
# [Annotator](https://public.md.ai/annotator/project/aGq4k6NW/workspace)
# - Lesson 3. RSNA Pneumonia detection using Kaggle data format [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-kaggle.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace)
# - Lesson 3. RSNA Pneumonia detection using MD.ai python client library [Github](https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb) [Annotator](https://public.md.ai/annotator/project/LxR6zdR2/workspace)
#
# *Copyright 2018 MD.ai, Inc.
# Licensed under the Apache License, Version 2.0*

# In[1]:


import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
import tensorflow as tf
from tensorflow import keras
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd
import glob
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ### First: Install Kaggle API for download competition data.

# In[2]:


DATA_DIR = "/home/mbenhamd/"

# Directory to save logs and trained model
ROOT_DIR = "/home/mbenhamd/working/"

# ###  MD.ai Annotator
#
# Additionally, If you are interested in augmenting the existing annotations, you can use the MD.ai annotator to view DICOM images, and create annotatios to be exported.
# MD.ai annotator project URL for the Kaggle dataset: https://public.md.ai/annotator/project/LxR6zdR2/workspace
#
# **Annotator features**
# - The annotator can be used to view DICOM images and create image and exam level annotations.
# - You can apply the annotator to filter by label, adjudicate annotations, and assign annotation tasks to your team.
# - Notebooks can be built directly within the annotator for rapid model development.
# - The data wrangling is abstracted away by the interface and by our MD.ai library.
# - Simplifies image annotation in order to widen the participation in the futrue of medical image deep learning.
#
# The annotator allows you to create initial annotations, build and run models, modify/finetune the annotations based on predicted values, and repeat.
# The MD.ai python client library implements functions to easily download images and annotations and to prepare the datasets used to train the model for classification. See the following example notebook for parsing annotations and training using MD.ai annotator:
# https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
# - MD.ai URL: https://www.md.ai
# - MD.ai documentation URL: https://docs.md.ai/

# ### Install Matterport's Mask-RCNN model from github.
# See the [Matterport's implementation of Mask-RCNN](https://github.com/matterport/Mask_RCNN).

# In[3]:


#get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')
os.chdir('Mask_RCNN')
# !python setup.py -q install


# In[4]:


# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# In[5]:


train_dicom_dir = os.path.join(DATA_DIR, 'train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'test_images')
train_label_csv = "/home/mbenhamd/stage_1_train_labels.csv"
det_class_path = '/home/mbenhamd/stage_1_detailed_class_info.csv'


# ### Some setup functions and classes for Mask-RCNN
#
# - dicom_fps is a list of the dicom image path and filenames
# - image_annotions is a dictionary of the annotations keyed by the filenames
# - parsing the dataset returns a list of the image filenames and the annotations dictionary

# In[6]:


def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir + '/' + '*.dcm')
    return list(set(dicom_fps))


def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


# In[7]:


# The following parameters have been selected to reduce running time for demonstration purposes
# These are not optimal

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    BACKBONE = 'resnet101'
    MEAN_PIXEL = np.array([63.1, 63.1, 63.1])
    NUM_CLASSES = 2  # background + 1 pneumonia classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MAX_INSTANCES = 100
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200

    MAX_GT_INSTANCES = 100
    USE_MINI_MASK = True
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    STEPS_PER_EPOCH = 1000
    TOP_DOWN_PYRAMID_SIZE = 64
    WEIGHT_DECAY = 1e-5
    LEARNING_RATE = 1e-3


config = DetectorConfig()
config.display()


# In[8]:


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


# ### Examine the annotation data, parse the dataset, and view dicom fields

# In[9]:


# training dataset
anns = pd.read_csv(train_label_csv)

# In[10]:


print(anns.count())

# In[11]:


image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

# In[12]:


#image_df = pd.DataFrame({'path': glob.glob(os.path.join(train_dicom_dir, '*.dcm'))})
#image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

# In[13]:


DCM_TAG_LIST = ['PatientAge', 'ViewPosition', 'PatientSex', 'PixelSpacing']


def get_tags(in_path):
    c_dicom = pydicom.read_file(in_path, stop_before_pixels=True)
    tag_dict = {c_tag: getattr(c_dicom, c_tag, '')
                for c_tag in DCM_TAG_LIST}
    tag_dict['path'] = in_path
    return pd.Series(tag_dict)


#image_meta_df = image_df.apply(lambda x: get_tags(x['path']), 1)
# show the summary
#image_meta_df['PatientAge'] = image_meta_df['PatientAge'].map(int)
#image_meta_df["path"] = image_meta_df["path"].map(lambda x: os.path.splitext(os.path.basename(x))[0])
#image_meta_df["PixelSpacing"] = image_meta_df["PixelSpacing"].map(lambda x: x[0])

# In[14]:


#set(image_meta_df["ViewPosition"])
# AP = 0
# PA = 1
#image_meta_df["ViewPosition"] = image_meta_df["ViewPosition"].map(lambda x: 0 if x == 'AP' else 1)

# In[15]:


#set(image_meta_df["PatientSex"])
# F = 0
# M = 1
#image_meta_df["PatientSex"] = image_meta_df["PatientSex"].map(lambda x: 0 if x == 'F' else 1)

# In[16]:


#image_meta_df = image_meta_df.rename(index=str, columns={"path": "patientId"})

# In[17]:


#class_info = pd.read_csv(det_class_path)

# In[18]:


#df_class_info = image_meta_df.join(class_info.set_index('patientId'), on='patientId')

# In[19]:


#final_df = df_class_info.join(anns.set_index('patientId'), on='patientId')

# In[20]:


# Lung Opacity = 0
# No Lung Opacity / Not Normal = 1
# Normal = 2
#set(final_df["class"])

# In[21]:


#final_df["class"] = final_df["class"].map(lambda x: 0 if x == 'Lung Opacity' else (2 if x == "Normal" else 1))

# In[22]:


#final_df

# In[23]:


#ds = pydicom.read_file(image_fps[10])  # read dicom image from filepath
#image = ds.pixel_array  # get image array

# In[24]:


# show dicom fields
#ds

# In[25]:


# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

# ### Split the data into training and validation datasets
# **Note: We have only used only a portion of the images for demonstration purposes. See comments below.**
#
#  - To use all the images do: image_fps_list = list(image_fps)
#  - Or change the number of images from 100 to a custom number

# In[26]:


######################################################################
# Modify this line to use more or fewer images for training/validation.
# To use all images, do: image_fps_list = list(image_fps)
image_fps_list = list(image_fps)
#####################################################################

# split dataset into training vs. validation dataset
# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
sorted(image_fps_list)
random.seed(42)
random.shuffle(image_fps_list)

validation_split = 0.1
split_index = int((1 - validation_split) * len(image_fps_list))

image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

print(len(image_fps_train), len(image_fps_val))

# ### Create and prepare the training dataset using the DetectorDataset class.

# In[27]:


# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()

# ### Let's look at a sample annotation. We see a bounding box with (x, y) of the the top left corner as well as the width and height.

# In[28]:


# Show annotation(s) for a DICOM image
test_fp = random.choice(image_fps_train)
image_annotations[test_fp]

# In[29]:


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

# ### Display a random image with bounding boxes

# In[30]:


# Load and display random samples and their bounding boxes
# Suggestion: Run this a few times to see different examples.

image_id = random.choice(dataset_train.image_ids)
image_fp = dataset_train.image_reference(image_id)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)

print(image.shape)
'''
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')
plt.axis('off')

print(image_fp)
print(class_ids)
'''
# In[31]:


model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

# In[32]:


# model.load_weights(model_path, by_name=True)


# ### Image Augmentation. Try finetuning some variables to custom values

# In[33]:


# Image augmentation
augmentation = iaa.SomeOf((0, 2), [
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    iaa.Affine(rotate=(-8, 8)),
    iaa.Affine(shear=(-8, 8)),
    iaa.Fliplr(1.0),
    iaa.Add((-10, 10)),
    iaa.Multiply((0.9, 1.10)),
    iaa.ContrastNormalization((0.95, 1.05)),
    iaa.Sharpen(alpha=(0.0, 0.2),
                lightness=(0.8, 1.2))
])

# ### Now it's time to train the model. Note that training even a basic model can take a few hours.
#
# Note: the following model is for demonstration purpose only. We have limited the training to one epoch, and have set nominal values for the Detector Configuration to reduce run-time.
#
# - dataset_train and dataset_val are derived from DetectorDataset
# - DetectorDataset loads images from image filenames and  masks from the annotation data
# - model is Mask-RCNN

# 500/500 [==============================] - 796s 2s/step - loss: 1.5425 - rpn_class_loss: 0.1004 - rpn_bbox_loss: 0.3746 - mrcnn_class_loss: 0.2041 - mrcnn_bbox_loss: 0.4207 - mrcnn_mask_loss: 0.4426 - val_loss: 1.4826 - val_rpn_class_loss: 0.0862 - val_rpn_bbox_loss: 0.3742 - val_mrcnn_class_loss: 0.1892 - val_mrcnn_bbox_loss: 0.4105 - val_mrcnn_mask_loss: 0.4226

# Epoch 5/5
# 500/500 [==============================] - 1084s 2s/step - loss: 1.6897 - rpn_class_loss: 0.1178 - rpn_bbox_loss: 0.4411 - mrcnn_class_loss: 0.1987 - mrcnn_bbox_loss: 0.4525 - mrcnn_mask_loss: 0.4796 - val_loss: 1.6887 - val_rpn_class_loss: 0.1233 - val_rpn_bbox_loss: 0.4545 - val_mrcnn_class_loss: 0.1918 - val_mrcnn_bbox_loss: 0.4482 - val_mrcnn_mask_loss: 0.4710

# In[34]:


NUM_EPOCHS = 10
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                                                 patience=2, verbose=1, cooldown=2, min_lr=0.00001)

# In[ ]:


# Train Mask-RCNN Model
import warnings

warnings.filterwarnings("ignore")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=NUM_EPOCHS,
            layers='all',
            augmentation=augmentation,
            )

# In[ ]:


# select trained model
dir_names = next(os.walk(model.model_dir))[1]
key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)

if not dir_names:
    import errno

    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(self.model_dir))

fps = []
# Pick last directory
for d in dir_names:
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))


# In[ ]:


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# ### How does the predicted box compared to the expected value? Let's use the validation dataset to check.
#
# Note that we trained only one epoch for **demonstration purposes ONLY**. You might be able to improve performance running more epochs.

# In[ ]:

'''
# Show few example of ground truth vs. predictions on the validation dataset
dataset = dataset_val
fig = plt.figure(figsize=(10, 30))

for i in range(4):
    image_id = random.choice(dataset.image_ids)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                       image_id, use_mini_mask=False)

    plt.subplot(6, 2, 2 * i + 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])

    plt.subplot(6, 2, 2 * i + 2)
    results = model.detect([original_image])  # , verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'],
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
'''
# In[ ]:


# Get filenames of test dataset DICOM images
test_image_fps = get_dicom_fps(test_dicom_dir)


# ### Final steps - Create the submission file

# In[ ]:


# Make predictions on test images, write out sample submission
def predict(image_fps, filepath='sample_submission.csv', min_conf=0.90):
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]

    with open(filepath, 'w') as file:
        file.write("patientId,PredictionString" + "\n")
        for image_id in tqdm(image_fps):
            ds = pydicom.read_file(image_id)
            image = ds.pixel_array

            # If grayscale. Convert to RGB for consistency.
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)

            patient_id = os.path.splitext(os.path.basename(image_id))[0]

            results = model.detect([image])
            r = results[0]

            out_str = ""
            out_str += patient_id
            out_str += ","
            assert (len(r['rois']) == len(r['class_ids']) == len(r['scores']))
            if len(r['rois']) == 0:
                pass
            else:
                num_instances = len(r['rois'])
                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        out_str += str(round(r['scores'][i], 2))
                        out_str += ' '

                        # x1, y1, width, height
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        bboxes_str = "{} {} {} {}".format(x1, y1, width, height)
                        out_str += bboxes_str
                        out_str += ' '

            file.write(out_str + "\n")


# In[ ]:


# predict
sample_submission_fp = '/home/mbenhamd/sample_submission.csv'
predict(test_image_fps, filepath=sample_submission_fp)

# In[ ]:


output = pd.read_csv(sample_submission_fp, names=['patientId', 'PredictionString'])
output.head(50)

# In[ ]:


(1000 - output["PredictionString"].isnull().sum()) / 100.0

