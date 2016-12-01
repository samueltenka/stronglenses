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

The code for the non-neural baselines can be found in the ipython notebooks folder of our git repo, inside TrySVM_RF_LogReg_Plus_Weight_Visualization.ipynb

### 0.1.0. Logistic Regression (logreg)

`Logistic regression` is equivalent to a densely connected
neural network without hidden layers, equipped with a sigmoid
activation function.
We performed logistic regression with exactly default Scikit-learn
settings, namely:

logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr',
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0)

We can visualize logistic regression weights:

![logistic regression channel 0 weights](/discussion/figures/log_reg_weight_visualizations_channel_0.png)

![logistic regression channel 1 weights](/discussion/figures/log_reg_weight_visualizations_channel_1.png)

![logistic regression channel 2 weights](/discussion/figures/log_reg_weight_visualizations_channel_2.png)

Interesting geometric patterns are apparent - some things Daniel thinks are important are the lack of weight symmetry across channels - particularly in how the the log reg deals with each central "blob" in the above images. The weights associate the blobs in channel 0 and 1 with being a strong lense, and associate the blobs in channel 2 with NOT being a strong lense. @nord said that the channel value inputs (which are unnormalized) contain additional information beyond simply relative light intensities. Could it be possible that the logreg is using this additional information to inform its decisions?

The logreg itself performed quite well - on a train/test split of 50/50, it achieved a testing AUROC of 0.987 and a testing accuracy of 0.944

### 0.1.1. Random Forests (rf)

A `random forest` is a weighted ensemble of `trees`, each of which
is a linear classifier on the space of axis-aligned threshold features
whose set of weights obeys the tree property, namely that every nonzero 
weight is dwarfed by all larger nonzero weights.
We trained a random forest with near-standard Scikit-learn settings,
namely

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

The resulting pattern of `feature importances` is quite interesting:

![random forest importances channel 0](/discussion/figures/rf_importance_visualizations_channel_0.png)

![random forest importances channel 1](/discussion/figures/rf_importance_visualizations_channel_1.png)

![random forest importances channel 2](/discussion/figures/rf_importance_visualizations_channel_2.png)

Daniel: The visualizations seem less cohesive than the logistic regression weights. They share some similar patterns (like an emphasis on the central portions of the image, for obvious reasons).
However, the patterns that random forests emphasizes surrounding the central portion differ from the logreg patterns.

The rf performed as well as the logreg, with an AUROC of 0.99 and an accuracy of 0.933

Sam: The above visualizations shows how important each feature is. Since the
features are the RGB bitmap values for each pixel (hence 64 x 64 x 3 in
number), we can visualize the importance map itself as a bitmap image.
Thus, a bright blue patch in the upper left corner would indicate that
the blueness of the astronomical image in the upper left corner is 
especially discriminative.

In particular, we are surprised that it seems rotation asymmetric.
Could this reveal a behavior of the simulator we had overlooked? 

### 0.1.2. Support Vector Machine (svc - the c is for classifier)

A `linear SVM` is a classifier that finds a linear boundary
that maximizes a weighted sum of accuracy and margin. Margin
is the smallest distance of a boundary to a correctly labeled point,
usually made robust to outliers via a softness parameter. 
We trained such an SVM with near-standard Scikit-learn settings,
namely

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='rbf', max_iter=-1, probability=True, random_state=None,
  shrinking=True, tol=0.001, verbose=False)

With standard parameters, the svc performed quite poorly.Hyperparameter adjustments are certainly needed.

The SVC obtained an AUROC of excactly 0.5 (rip) and and accuracy of 0.497666666667 (double rip)

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

### 0.1.2. Simple Convolution Network (simpCNN)

The first neural network model to generate admirable results was a simple convolution neural network, consisting of 3 convolutional layers, a max pooling layer, and then 3 densely connected layers, with the final layer using softmax activation instead of softplus. All layers except for the final layer use softplus activation, and the adam optimizer was used.

### 0.1.3. Shallow Residual Net (SHALLOW_RES)

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
                
                #parameters    Slowness     Test Loss   Test Acc    Test Acc @ 80% Yield
                (Millions)  (secs / epoch)  (nits)      (%)         (%)
    RANDFOR         0.001       N/A         ?.???       93.3        ??.?
    SVM             ?.??        N/A          N/A        ??.?        ??.?
    MLP             0.79         10         0.285       94.8        96.8 
    MLP_WIDE        3.16         25         0.233       95.5        97.2
    LOGISTIC        0.01          1         0.148       95.2        98.9
    SHALLOW_RES     0.03         20         0.124       95.7        99.5
    SOFTPLUS_3      0.13         15         0.096       97.4        99.0
    SQUEEZE_SKIP    0.07         10         0.065 *     97.8 *     100.0 *

The shallow residual network significantly outperforms (as expected)
the multilayer perceptrons: its loss is about half of the wide MLP,
achieved using two orders of magnitude fewer parameters, and in 80%
of the train time. The neural networks perform ?? relative to our 
non-neural baselines.  

Of course, the above comparison is fair only if all three models have converged.
As the following plot shows, the nets seem indeed to have neared convergence:

![training histories](/discussion/autogenfigures/LOGISTIC_vs_SHALLOW_RES_vs_SQUEEZE_SKIP.hist.png)

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
max(p, 1-p) of the predicted probability distribution.
Specifically, we plot the accuracy (of the model on the datapoints on which the
model is most confident) vs the number of datapoints we require.

![yield curves](/discussion/autogenfigures/LOGISTIC_vs_SHALLOW_RES_vs_SQUEEZE_SKIP.yield.png)

The curves are nonincreasing (save for sampling error), as they should be.
We see that SHALLOW_RES outperforms the MLP's by 1 to 3 accuracy points on
yields less than 0.8. Furthermore, for yields less than 0.8, SHALLOW_RES
attains `99.5%` accuracy, a figure that we find pleasing.

Note: Sam has always found measures such as AUC arbitrary and hence
unpersuasive. For instance, AUC has no unique maximum, is blind to
order-preserving distortions of probability, and is not easily tuneable
to penalize false positives and false negatives unequally. Yet because
AUC is standard, we present it:

![roc curves](/discussion/autogenfigures/LOGISTIC_vs_SHALLOW_RES_vs_SQUEEZE_SKIP.roc.png)

On a log scale, the differences between algorithms become more apparent:

![log roc curves](/discussion/autogenfigures/LOGISTIC_vs_SHALLOW_RES_vs_SQUEEZE_SKIP.logroc.png)
