# CSCE-636-Video-Classification

training.py
- for model training
- the output is "weights.hdf5" which contains all the weights of the trained model

testing.py
- for model testing
- the output is the json file + a figure. one json file+figure per testing video

trainlist.txt
- list of training videos
- user must put in the names of all training videos in here before training the model

testlist.txt
- list of testing videos
- user must put in the names of all testing videos in here before testing the model


# Instructions on How to Test the Trained DNN

Install dependencies:
- Python 2.7
- Keras
- Tensorflow
- OpenCV
- Scipy, sklearn, skimage, glob, tqdm

How to train:
- Put the name of the training videos in the trainlist.txt file
- Put the training video files in the training_videos folder
- Start the training process using command: python training.py

How to test:
- Put the name of the testing videos in the testlist.txt file
- Put the testing video files in the testing_videos folder
- Run the test using command: python testing.py

NOTE:
- Since the weights.hdf5 exceeds the file size limit of GitHub, we cannot push it to the repo. 
- Thus, user can either run the training code to generate the weight file for testing or download it at this link (most updated version):
  https://drive.google.com/file/d/1J0eJH_90PYdXjSeLwyf1AiaBw_L-XPfU/view?usp=sharing
