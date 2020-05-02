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

Test Videos:
- there are 5 testing videos already inside the testing_videos folder
- other 5 videos on YouTube:
  - https://youtu.be/mwPCg87oioQ
  - https://youtu.be/cCALID3jtZY
  - https://youtu.be/SYdYS5372ag
  - https://youtu.be/VgoHuu2Zl1M
  - https://youtu.be/-CIRVqJtBrE
  

# Instructions on How to Train and Test the DNN

Install dependencies:
- Python 2.7
- Keras
- Tensorflow
- OpenCV
- Scipy, sklearn, skimage, glob, tqdm

How to train:
- Put the full name of the training videos in the trainlist.txt file
- Put the training video files in the training_videos folder
- Start the training process using command: python training.py

How to test:
- Put the full name of the testing videos in the testlist.txt file 
- Put the testing video files in the testing_videos folder
- Run the test using command: python testing.py

NOTE:
- Link of the weights.hdf5 file of submission 8 is here:
  https://drive.google.com/file/d/1xIl63cJKruDWNKPd3jSYznw_L4Ly27Md/view?usp=sharing
- Old weight files for testing can be downloaded it at this link:
  https://drive.google.com/open?id=19J8olymAioX9sbiVs7yulqMCNvNc3Xvh
- Put the weights.hdf5 file in the same folder as the testing.py file

Instruction Videos on how to use this code:
- Training: https://youtu.be/RKVa47go7N0
- Testing: https://youtu.be/gmvvhNZbfjA
