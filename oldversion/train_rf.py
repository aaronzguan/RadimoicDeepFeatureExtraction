import pickle
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from classifier_evaluator import classifier_evaluator
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import h5py



##############################
# train_rf.py  is used to train the random forest given by
# the features from training samples
##############################

SVM_support = True
RF_support = True

# ResNet
# x = loadmat('features/deepFeature_res')['Feature']
# y = loadmat('features/deepFeature_label_res')['label']
# print('ResNet-50 Feature Loaded')

# AlexNet
# x = pd.read_csv('features/deepFeature.csv', index_col=0)
# y = pd.read_csv('features/deepFeature_label.csv', index_col=0)
# print('AlexNet Feature Loaded')
# x_test = pd.read_csv('features/deepFeature_test.csv', index_col=0)
# print('testing data loaded')
# y_test = pd.read_csv('features/deepFeature_test_label.csv', index_col=0)
# print('testing label loaded')

#VGG
filepath = 'features/deepFeature_vgg.mat'
array = {}
f = h5py.File(filepath)
for k, v in f.items():
    array[k] = np.array(v).transpose()

x = array['Feature']

y = loadmat('features/deepFeature_label_vgg')['label']
print('VGG-16 Feature Loaded')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)

# print('Number of training features: ' + str(len(y_train)))
# print('Number of testing features: ' + str(len(y_test)))


if SVM_support:
    print('')
    print('SVM Classifier')
    print('')
    SVM = svm.SVC()
    SVM.fit(x_train, y_train)

    y_pred_class = SVM.predict(x_test)
    Evaluation = classifier_evaluator(y_test, y_pred_class)
    Evaluation.accuracy()
    Evaluation.confusion_matrix()

    # save classifier
    classifier_svm = open('SVM_vgg.pkl', 'wb')
    pickle.dump(SVM, classifier_svm)
    classifier_svm.close()

# Implementing the random forest classifier
if RF_support:
    print('')
    print('Random Forest Classifier')
    print('')
    RF = RandomForestClassifier(n_estimators=200, oob_score=True)
    RF.fit(x_train, y_train)
    print('Start testing')

    y_pred_class = RF.predict(x_test)
    Evaluation = classifier_evaluator(y_test, y_pred_class)
    Evaluation.accuracy()
    Evaluation.confusion_matrix()

    # save classifier
    classifier_rf = open('RandomForest_vgg.pkl', 'wb')
    pickle.dump(RF, classifier_rf)
    classifier_rf.close()