#!/usr/bin/env bash

pip install --upgrade pip

DATA_DIR_LOC='../../data'
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    pip install kaggle --upgrade
    kaggle competitions download -c global-wheat-detection
    unzip global-wheat-detection.zip
    rm global-wheat-detection.zip  
fi
echo $(pwd)