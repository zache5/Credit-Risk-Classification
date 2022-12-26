# Credit-Risk-Classification
This code is traning a Logistic Regression model with given data with both a normal set of data, as well as an oversampled set of data. This is a good test to see how the model reacts when it is given oversampled data compared to normal data, because the results: recoil, accuracy, etc may vary. There are many other types of sampling techniques that we can use to compare models and organize our data, and so it is important to run tests like this one to see which specifc method is the best for your given dataset.

## Technologies
In this Code, I am using Numpy, Pandas, Pathlib, and multiple packages from sklearn.metrics: balanced_accuracy_score, confusion_matrix, and classification_report_imbalanced

## Installation Guide
import numpy as np,
import pandas as pd,
from pathlib import Path,
from sklearn.metrics import balanced_accuracy_score,
from sklearn.metrics import confusion_matrix,
from imblearn.metrics import classification_report_imbalanced

## Usage
We are able to print out and see the confusion matrix for both the normal and the oversampled data. The oversampled data happened to perform better, with a higher recoil and accuracy score, but we can continue to run these tests with different sampling methods and parameters in order to get the best score possible.

## Contributors
I am the main contributor for this project!
