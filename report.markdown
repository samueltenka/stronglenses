# Model Results (2016-11-21) 
    author: sam tenka

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

### 0.1.1. Random Forests (RF)

### 0.1.2. Support Vector Machine (SVM)

## 0.1. Neural Networks

We also tried three simple neural network architectures. We trained from scratch, 
but aim to do transfer learning soon.

### 0.1.0. Multilayer Perceptron (MLP)

0.79 M parameters.

### 0.1.1. Multilayer Perceptron Wide (MLP_WIDE)

3.16 M parameters.

### 0.1.2. Shallow Residual Net (SHALLOW_RES)

0.03 M parameters.

# 1. Model Performance 

We train each neural network for 60 epochs, and present the resulting metrics
(binary crossentropy and accuracy), computed on a withheld test set:
                
                #parameters     Speed       Test Loss   Test Accuracy
                (Millions)  (Epochs / s)    (nits)      (nits)
    MLP             0.79        ~10         0.2849      0.9444
    MLP_WIDE        3.16        ~25         0.2332      0.9512
    SHALLOW_RES     0.03        ~20         0.1244      0.9526

The shallow residual network significantly outperforms (as expected)
the multilayer perceptrons: its loss is about half of the wide MLP,
achieved using two orders of magnitude fewer parameters, and in 80%
of the train time.

Now, the above comparison is fair only if all three models have converged.
As the following plot shows, the nets seem indeed to have neared convergence:
![hi](/discussion/figures/MLP_vs_MLP_WIDE_vs_SHALLOW_RES.hist.png)
Observe the overfitting (in which train loss << validation loss) in the 
multilayer percerptrons. Our shallow residual network, on the other hand,
has much fewer parameters, and is further regularized with batch normalization
and dropout, so it has, in fact, the opposite relation: train_loss >> validation_loss.  
In any case, it is apparent that further training is unlikely to change
the outperformance of MLP's by the shallow residual network.

Now, Dr. Nord suggested that a classifier with high precision but low
recall would still be valuable, and even better a classifier that
knew when it was sure or unsure. We thus investigate the relation
of accuracy to our (probabilistic) models' confidences, computed as the maximum
max(p, 1-p) of the predicted probability distribution, with their accuracies.
Specifically, we plot the accuracy (of the model on the datapoints on which the
model is most confident) vs the number of datapoints we require.
![hi](/discussion/figures/MLP_vs_MLP_WIDE_vs_SHALLOW_RES.yield.png)
The curves are nonincreasing (save for sampling error), as they should be.
We see that SHALLOW_RES outperforms the MLP's by 1 to 3 accuracy points on
yields less than 0.8. Furthermore, for yields less than 0.8, SHALLOW_RES
attains `99.5%` accuracy, a figure that we find pleasing.

Note: we have always found measures such as AUC arbitrary and hence
unpersuasive.

