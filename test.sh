#!/bin/bash

# Replace the variables with your github repo url, repo name, test
# video name, json named by your UIN
GIT_REPO_URL="https://github.com/phatnguyen0430/CSCE-636-Video-Classification.git"
REPO="CSCE-636-Video-Classification"
VIDEO_FILE="testlist.txt"
UIN_JSON="726006842.json"
UIN_JPG="726006842.jpg"
git clone $GIT_REPO_URL
cd $REPO

wget "https://drive.google.com/open?id=1xIl63cJKruDWNKPd3jSYznw_L4Ly27Md"
# Replace this line with commands for running your test python file.
echo $VIDEO_FILE
python testing.py

# If your test file is ipython file, uncomment the following lines and
# replace IPYTHON_NAME with your test ipython file.
# IPYTHON_NAME="test.ipynb"
# echo $IPYTHON_NAME
# jupyter notebook
# rename the generated timeLabel.json and figure with your UIN.

# cp timeLabel.json $UIN_JSON
# cp timeLabel.jpg $UIN_JPG