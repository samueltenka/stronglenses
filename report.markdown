# Model Results (2016-11-21) 
    author: sam tenka, daniel zhang

    date: 2016-11-21

    descr: Overview of networks trained 2016-11-21.

# 0. Model Descriptions

We are posed an image-classification task.

## 0.0. Non-Neural Baselines

We tried 3 out-of-the box classifiers. Unlike convnets, these do not explicitly
account for the spatial structure of images. Specifically, they treat images as
size-12288 vectors (since 4096 pixels x 3 numbers per pixel = 12288 numbers).
Thus, these models are unable to leverage the notion of `neighboring pixels`.
By adding an appropriate "regularization term", we could encode that notion,
but this is not, for us, a high priority.

### 0.1.0. Logistic Regression (LR)

`Logistic regression` is equivalent to a densely connected
neural network without hidden layers, equipped with sigmoid
activation function.
We performed logistic regression with near-standard Scikit-learn
settings, namely ???.
We can visualize logistic regression weights: 
???

### 0.1.1. Random Forests (RF)

A `random forest` is a weighted ensemble of `trees`, each of which
is a linear classifier on the space of axis-aligned threshold features
whose set of weights obeys the tree property, namely that every nonzero 
weight is dwarfed by all larger nonzero weights.
We trained a random forest with near-standard Scikit-learn settings,
namely ???.
The resulting pattern of `feature importances` is quite interesting:

![random forest importances](/discussion/figures/rf_importances.png)

In particular, we are surprised that it seems rotation asymmetric.
Could this reveal a behavior of the simulator we had overlooked? 

### 0.1.2. Support Vector Machine (SVM)

A `linear SVM` is a classifier that finds a linear boundary
that maximizes a weighted sum of accuracy and margin. Margin
is the smallest distance of a boundary to a correctly labeled point,
usually made robust to outliers via a softness parameter. 
We trained such an SVM with near-standard Scikit-learn settings,
namely ???.
The resultant weights are:
???

## 0.1. Neural Networks

We also tried three simple neural network architectures. We trained from scratch, 
but aim to do transfer learning soon.

### 0.1.0. Multilayer Perceptron (MLP)

Our MLP is a sequence of densely connected softplus layers
each of whose inputs is regularized by 50% dropout. We have 2
hidden layers of sizes 64 and 64, yielding 0.79 M parameters.

### 0.1.1. Multilayer Perceptron Wide (MLP_WIDE)

MLP Wide is an MLP modified to have hidden layers of
sizes 256 and 64, yielding 3.16 M parameters.

### 0.1.2. Shallow Residual Net (SHALLOW_RES)

A residual block we define as follows: take a
linear convolutional layers whose output is
regularized by batch normalization, then fed
through a ReLU activation. Concatenate two such
layers, then sum with the original input. The
sum is a special type of `skip connection`, and
allows gradients to travel across shorter paths 
as they backpropagate from one layer to another.
To complete our residual block, we maxpool that sum.

To build SHALLOW_RES, we feed the input through one
linear convolution layer, then a maxpool, thereby
compressing 12288 features into  ~ 1230 features.
We then apply 1 residual block. We use 8 channels,
kernel 3x3, maxpool 3x3 throughout. After the convolutional
sections, we apply a sigmoid MLP with parameters as in
0.1.0, for a total of 0.03 M parameters.

# 1. Model Performance 

We train each neural network for 60 epochs, and present the resulting metrics
(binary crossentropy and accuracy), computed on a withheld test set:
                
                #parameters     Speed       Test Loss   Test Acc    Test Acc @ 80% Yield
                (Millions)  (Epochs / s)    (nits)      (%)         (%)
    LOGREG          ?.??        ~??         ?.???       ??.?        ??.?
    RANDFOR         ?.??        ~??         ?.???       ??.?        ??.?
    SVM             ?.??        N/A          N/A        ??.?        ??.?
    MLP             0.79        ~10         0.285       94.4        96.9 
    MLP_WIDE        3.16        ~25         0.233       95.1        97.3
    SHALLOW_RES     0.03        ~20         0.124       95.3        99.5

The shallow residual network significantly outperforms (as expected)
the multilayer perceptrons: its loss is about half of the wide MLP,
achieved using two orders of magnitude fewer parameters, and in 80%
of the train time. The neural networks perform ?? relative to our 
non-neural baselines.  

Of course, the above comparison is fair only if all three models have converged.
As the following plot shows, the nets seem indeed to have neared convergence:

![net histories](/discussion/figures/MLP_vs_MLP_WIDE_vs_SHALLOW_RES.hist.png)

Observe the overfitting (in which train loss << validation loss) in the 
multilayer percerptrons. Our shallow residual network, on the other hand,
has much fewer parameters, and is further regularized with batch normalization
and dropout, so it has, in fact, the opposite relation: train_loss > validation_loss. 
In any case, it is apparent that further training is unlikely to change
the outperformance of MLP's by the shallow residual network. Note that
the rest of this document is based on the model of least validation loss
(not test loss) from a whole training history; in other words, we use 
a pocket algorithm to learn despite noisy training.

Now, Dr. Nord suggested that a classifier with high precision but low
recall would still be valuable, and even better a classifier that
knew when it was sure or unsure. We thus investigate the relation
of accuracy to our (probabilistic) models' confidences, computed as the maximum
max(p, 1-p) of the predicted probability distribution, with their accuracies.
Specifically, we plot the accuracy (of the model on the datapoints on which the
model is most confident) vs the number of datapoints we require.

![net yield curves](/discussion/figures/MLP_vs_MLP_WIDE_vs_SHALLOW_RES.yield.png)

The curves are nonincreasing (save for sampling error), as they should be.
We see that SHALLOW_RES outperforms the MLP's by 1 to 3 accuracy points on
yields less than 0.8. Furthermore, for yields less than 0.8, SHALLOW_RES
attains `99.5%` accuracy, a figure that we find pleasing.

Note: Sam has always found measures such as AUC arbitrary and hence
unpersuasive.

