from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image

from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

import random

data = []
labels = []

img_dims = (96 , 96, 3)


image_files = [f for f in glob.glob(r'facedataset'+"/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (96,96))
    image = img_to_array(image)
    data.append(image)
    
    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append(label)
    
#pre-processing
data = np.array(data, dtype='float')/255.0
labels = np.array(labels)

        
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


aug = ImageDataGenerator(
    rotation_range = 25,
    width_shift_range = 0.1,
    height_shift_range = 0.1, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# model build 



