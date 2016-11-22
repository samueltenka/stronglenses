# Model Results (2016-11-21) 
    author: sam tenka

    date: 2016-11-21

    descr: Overview of networks trained 2016-11-21.

# 0. Model Descriptions

## 0.0. Non-Neural Baselines

### 0.1.0. Logistic Regression (LR)

### 0.1.1. Random Forests (RF)

### 0.1.2. Support Vector Machine (SVM)

## 0.1. Neural Networks

### 0.1.0. Multilayer Perceptron (MLP)

0.79 M parameters.

### 0.1.1. Multilayer Perceptron Wide (MLP_WIDE)

3.16 M parameters.

### 0.1.2. Shallow Residual Net (SHALLOW_RES)

0.03 M parameters.

# 1. Model Performance 

We train each neural network for 60 epochs, and present the resulting test metrics (binary crossentropy and accuracy).

To further investigate network   
We also compare training speeds by listing time-per-training-epoch, and we demonstrate convergence by plotting loss against epoch.  

!(/discussion/figures/MLP.hist.png)

## 1.0. Multilayer Perceptron

After 60 epochs (~10 seconds each) of training, MLP achieves:

    loss=0.2849, acc=0.9444

## 1.1. Multilayer Perceptron --- Wide

After 60 epochs (~25 seconds each) of training, MLP_WIDE achieves:

    loss=0.2332, acc=0.9512

## 1.2. Shallow Resnet 

After ~60 epochs (~20 seconds each) of training, MLP_WIDE achieves:

    loss=0.1244, acc=0.9526

# 2. Discussion


