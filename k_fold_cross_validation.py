import numpy as np

"""
Cross-validation is a resampling procedure used to evaluate
machine learning models on a limited data sample.

Cross-validation is primarily used in applied machine learning
to estimate the skill of a machine learning model on unseen data.
That is, to use a limited sample in order to estimate how the model
is expected to perform in general when used to make predictions
on data not used during the training of the model.

When a specific value for k is chosen, it may be used in place of k in the
reference to the model, such as k = 10 becoming 10-fold cross-validation.
k is the number of groups that a given data sample is to be split into.

A poorly chosen value for k may result in a mis-representative idea of the
skill of the model, such as a score with a high variance (that may change
a lot based on the data used to fit the model), or a high bias, (such as
an overestimate of the skill of the model).

It generally results in a less biased or less optimistic estimate of
the model skill than other methods, such as a simple train/test split.

1.Shuffle the dataset randomly.
2.Split the dataset into k groups
3.For each unique group:
    Take the group as a hold out or test data set
    Take the remaining groups as a training data set
    Fit a model on the training set and evaluate it on the test set
    Retain the evaluation score and discard the model
4.Summarize the skill of the model using the sample of model evaluation scores
"""

# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold

# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# prepare cross validation
# create an instance that splits a dataset into 3 folds,
# shuffles prior to the split,
# and uses a value of 1 for the pseudorandom number generator.
kfold = KFold(3, True, 1)

# enumerate splits
for train, test in kfold.split(data):
    print (train, test)
    print('train: %s, test: %s' % (data[train], data[test]))
    
