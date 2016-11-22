# stronglenses
    author: sam tenka

    date: 2016-11-21

    descr: Strong Lens Detection via Neural Net.

           We'll classify deepfield astronomical
           images by whether or not they contain
           a strong gravitational lens. This project
           is part of the Michigan Data Science
           Team's 2016 activities.

# 0. Model Descriptions

# 1. Model Performance 

We train each model for 20 epochs, and present the resulting test metrics (binary crossentropy and accuracy).
To further investigate network   
We also compare training speeds by listing time-per-training-epoch, and we demonstrate convergence by plotting loss against epoch.  


## 1.0. Multilayer Perceptron

After 20 epochs (~10 seconds each) of training, MLP achieves test metrics:

    loss=0.1557, acc=0.9512

## 1.1. Multilayer Perceptron --- Wide

After 20 epochs (~25 seconds each) of training, MLP_WIDE achieves training metrics:

    loss=0.4715, acc=0.9310

## 1.2. Shallow Resnet 

After ~20 epochs (~20 seconds each) of training, MLP_WIDE achieves training metrics:

    loss=0.1752, acc=0.9293

# 2. Discussion


