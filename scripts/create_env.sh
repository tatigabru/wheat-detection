#!/bin/bash
conda init bash
conda update -y conda
conda create -y -n wheat python=3.7
conda activate wheat
pip install --upgrade pip
pip install -r requirements.txt