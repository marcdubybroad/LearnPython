#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# import the sklear gaussian library
from sklearn.naive_bayes import GaussianNB

# create the classifier
classifier = GaussianNB()

# train the fit
classifier.fit(features_train, labels_train)

# classify
prediction = classifier.predict(features_test)

# get the accuracy
from sklearn.metrics import accuracy_score
acuracy = accuracy_score(prediction, labels_test)

# print
print "the author accuracy is: ", acuracy

#########################################################


