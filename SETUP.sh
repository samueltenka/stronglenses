#!/bin/bash

# basic development tools helpful not necessary to develop Strong Lenses 
sudo apt-get install tmux
sudo apt-get install vim

# python: numerics and machine learning basics
sudo apt-get install python-numpy
sudo apt-get install python-sklearn

# tensorflow, a computation-graph compiler for Python:
sudo apt-get install python-pip
sudo apt-get install python-dev
sudo apt-get install python-virtualenv
virtualenv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
pip uninstall protobuf
pip install --upgrade $TF_BINARY_URL

# keras
sudo pip install keras
pip install -U numpy 
sudo apt-get install cython
sudo apt-get install python-h5py
pip install cuDNN

# mkdir ../mdst
# mkdir ../mdst/sldata
# mkdir ../mdst/slcheckpoints

