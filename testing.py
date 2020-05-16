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

from sklearn.metrics import accuracy_score

import json
# ==================================================================================== #

base_model = VGG16(weights='imagenet', include_top=False)

# create the model (identical to the one used in training)
# tested from 3-6 layers, 32-1024 neurons/layer
model = Sequential()

model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# model.add(Dense(128, activation='relu', input_shape=(25088,)))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))

# model.add(Dense(512, activation='relu', input_shape=(25088,)))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))

# load and compile the trained weights
model.load_weights("weights-sub8.hdf5")
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# ==================================================================================== #

# read the test list
f = open("given-videos/testlist.txt", "r")
temp = f.read()
videos = temp.split('\n')

# create the test dataframe
test = pd.DataFrame()
test['video_name'] = videos
test = test[:-1]
test_videos = test['video_name']
test.head()

# ==================================================================================== #

# creating two lists to store predicted and actual tags
num_total_correct = 0
num_frames = 0

images_list = []
classes_list = []
predictions_list = []
correctness_list = []
probability_list = []

# for loop to extract frames from each test video
for i in tqdm(range(test_videos.shape[0])):
    data_outfile = []
    data_outfile.append("heads up: detecting mopping floor and washing dishes activity")
    count = 0
    videoFile = test_videos[i]

    print("\n\nVideo File: " + videoFile)

    cap = cv2.VideoCapture('given-videos/' + videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    timestamps = []
    # removing all other files from the extracted_frames folder
    files = glob('given-videos/*.jpg')
    for f in files:
        os.remove(f)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            # storing the frames of this particular video in extracted_frames folder
            filename ='given-videos/' + videoFile+ "_frame%03d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()

    # reading all the frames from extracted_frames folder
    images = sorted(glob("given-videos/*.jpg"))

    test_image = []
    test_class = []
    for i in tqdm(range(len(images))):
        # creating the image name
        test_image.append(images[i].split('/')[1])
        images_list.append(images[i].split('/')[1])
        # creating the class of image
        if images[i].find('MoppingFloor') != -1:
            test_class.append("housecleaning")
            classes_list.append("housecleaning")

        elif images[i].find('WashingDishes') != -1:
            test_class.append("washingdishes")
            classes_list.append("washingdishes")
        
        else:
            test_class.append("not_housework")
            classes_list.append("not_housework")
        
        # if images[i].find('MoppingFloor') != -1 or images[i].find('WashingDishes') != -1:
        #     test_class.append("housework")
        #     classes_list.append("housework")
        # else:
        #     test_class.append("not_housework")
        #     classes_list.append("not_housework")

    # store the images and their class in a dataframe
    test_data = pd.DataFrame()
    test_data['image'] = test_image
    test_data['class'] = test_class

    # convert the dataframe into csv file 
    test_data.to_csv('testing_frames_list.csv',header=True, index=False)

    # ==================================================================================== #

    # load the frame list
    test = pd.read_csv('testing_frames_list.csv')
    test.head()

    actual = test['class']
    predictions = []

    # extract video frames and make prediction
    for i in tqdm(range(test.shape[0])):
        # loading the image and keeping the target size as (224,224,3)
        img = image.load_img('given-videos/'+test['image'][i], target_size=(224,224,3))
        # converting it to array
        img = image.img_to_array(img)
        # normalizing the pixel value
        img = img/255
        
        # preprocess with VGG-16 base model and reshape
        test_image = []
        test_image.append(img)
        x_test = np.array(test_image)
        x_test = base_model.predict(x_test)
        x_test = x_test.reshape(x_test.shape[0], 7*7*512)

        # make prediction using our trained model
        prediction = model.predict_classes(x_test)      # 0 == housecleaning, 1 == not_housework, 2 == washingdishes
        probability = model.predict_proba(x_test)       # [1,0,0] == housecleaning, [0,1,0] == not_housework, [0,0,1] == washingdishes
        predictions.append(1.0 - probability[0][1])     # probability[0][1] == probability of not_housework
        if prediction == 0 and actual[i] == "housecleaning":
            num_total_correct += 1
            correctness_list.append('yes')
        elif prediction == 1 and actual[i] == "not_housework":
            num_total_correct += 1
            correctness_list.append('yes')
        elif prediction == 2 and actual[i] == "washingdishes":
            num_total_correct += 1
            correctness_list.append('yes')
        else:
            correctness_list.append('no')

        # if prediction == 0 and actual[i] == "housework":
        #     num_total_correct += 1
        #     correctness_list.append('yes')
        # elif prediction == 1 and actual[i] == "not_housework":
        #     num_total_correct += 1
        #     correctness_list.append('yes')
        # else:
        #     correctness_list.append('no')

        num_frames += 1
        predictions_list.append(prediction)
        probability_list.append(1.0 - probability[0][1])

    # combine time/label data and write out to JSON file
    for i in range(len(timestamps)):
        data_outfile.append([str(timestamps[i]), str(predictions[i])])
    plt.ylim(-0.25,1.25)
    plt.plot(timestamps, predictions)
    plt.savefig(videoFile + '.png')
    plt.close()

    with open(videoFile + '.json', 'w') as outfile:
        json.dump(data_outfile, outfile)

result = pd.DataFrame()
result['image']       = images_list
result['class']       = classes_list
result['prediction']  = predictions_list
result['probability'] = probability_list
result['correct']     = correctness_list

result.to_csv('testing_results.csv',header=True, index=False)

print("Total num frames: ", num_frames)
print("Correctly predicted:", num_total_correct)
print("Test Accuracy: ", float(num_total_correct)/num_frames*100)

# clean up all images from the extracted_frames folder
# files = glob('given-videos/*.jpg')
# for f in files:
#     os.remove(f)