import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from scipy import stats as s

# ==================================================================================== #
def create_model(config):
    model = Sequential()
    model.add(Dense(config[0], activation='relu', input_shape=(25088,)))
    for i in range(1, len(config)):
        model.add(Dropout(0.5))
        model.add(Dense(config[i], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    return model

# open the file containing the list of training videos
f = open("trainlist.txt", "r")
temp = f.read()
videos = temp.split('\n')

# create a dataframe
train = pd.DataFrame()
train['video_name'] = videos
train = train[:-1]
train.head()

# ==================================================================================== #

# create tags for training videos
train_video_tag = []
for i in range(train.shape[0]):
    train_video_tag.append(train['video_name'][i].split('/')[0])
    
train['tag'] = train_video_tag

# ==================================================================================== #

# remove old frames in the extracted_frames folder
files = glob('training_videos/extracted_frames/*')
for f in files:
    os.remove(f)

# extract the frames from training videos
for i in tqdm(range(train.shape[0])):
    count = 0
    videoFile = train['video_name'][i]
    cap = cv2.VideoCapture('training_videos/' + videoFile.split(' ')[0].split('/')[1])   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0): # get one frame per second
            # storing the frames in a new folder named extracted_frames
            filename ='training_videos/extracted_frames/' + videoFile.split('/')[1].split(' ')[0] +"_frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()

# ==================================================================================== #

# get label for all images
images = glob("training_videos/extracted_frames/*.jpg")
train_image = []
train_class = []
for i in tqdm(range(len(images))):
    # creating the image name
    train_image.append(images[i].split('/')[2])
    
    # creating the class of image
    if images[i].split('/')[2].split('_')[1] == "MoppingFloor" or images[i].split('/')[2].split('_')[1] == "WashingDishes":
        train_class.append("housework")
    else:
        train_class.append("not_housework")
    
# storing the images and their class in a dataframe
train_data = pd.DataFrame()
train_data['image'] = train_image
train_data['class'] = train_class

# convert the dataframe into csv file 
train_data.to_csv('training_frames_list.csv',header=True, index=False)

# ==================================================================================== #

# load the frame list
train = pd.read_csv('training_frames_list.csv')
train.head()

# create an empty list
train_image = []

# load all video frames
for i in tqdm(range(train.shape[0])):
    # load the image and keep the target size as (224,224,3)
    img = image.load_img('training_videos/extracted_frames/'+train['image'][i], target_size=(224,224,3))
    # convert it into an array
    img = image.img_to_array(img)
    # normalize the pixel value
    img = img/255
    # append the image to the train_image list
    train_image.append(img)
    
# convert the list to numpy array
x = np.array(train_image)

# shape of the array
print "\nRaw data shape: ", 
print(x.shape)

# ==================================================================================== #

# split the videos into training and validation set
y = train['class']
x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=42, test_size=0.2, stratify = y)

# create dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_validate = pd.get_dummies(y_validate)

print "y_train shape:    ", 
print(y_train.shape)
print "y_validate shape: ", 
print(y_validate.shape)

# ==================================================================================== #

print "Processing training data through VGG16 ...", 

# create the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# extract features for training frames
x_train = base_model.predict(x_train)

# extract features for validation frames
x_validate = base_model.predict(x_validate)

print("Done")
print "x_train shape:    ", 
print(x_train.shape)
print "x_validate shape: ", 
print(x_validate.shape)

# ==================================================================================== #

print "Reshaping training data for the final fully connected neural network ... ",

# reshape the training as well as validation frames in single dimension
x_train = x_train.reshape(x_train.shape[0], 7*7*512)
x_validate = x_validate.reshape(x_validate.shape[0], 7*7*512)

# normalize the pixel values
max = x_train.max()
x_train = x_train/max
x_validate = x_validate/max

print("Done")
print "x_train shape:    ", 
print(x_train.shape)
print "x_validate shape: ", 
print(x_validate.shape)
print("")

# ==================================================================================== #

# create the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

# ==================================================================================== #

# create a checkpoint file to store the trained weights 
mcp_save = ModelCheckpoint('weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compile the model
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, epochs=15, validation_data=(x_validate, y_validate), callbacks=[mcp_save], batch_size=128)

# remove used frames in the extracted_frames folder
files = glob('training_videos/extracted_frames/*')
for f in files:
    os.remove(f)