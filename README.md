# CSCE-636-Video-Classification

training.py
- for model training
- the output is "weights.hdf5" which contains all the weights of the trained model

test.py
- for model testing
- the output is the json file. one json file for each testing video

trainlist.txt
- list of training videos
- user must put in the names of all training videos in here before training the model

testlist.txt
- list of testing videos
- user must put in the names of all testing videos in here before testing the model

NOTE:
- since the weights.hdf5 exceeds the file size limit of GitHub, we cannot push it to the repo. 
- thus, user must run the training code before runnning the testing code
