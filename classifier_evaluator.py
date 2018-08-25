from sklearn import metrics
##############################
# classifier_evaluator.py is used to evaluate the classifier performance
##############################
class classifier_evaluator():
    def __init__(self, y_test, y_pred_class):
        self.y_test = y_test
        self.y_pred_class = y_pred_class

    def accuracy(self):
        print('Accuracy: ', metrics.accuracy_score(self.y_test, self.y_pred_class))

    def confusion_matrix(self):
        confusion = metrics.confusion_matrix(self.y_test, self.y_pred_class)
        TP = confusion[1,1]
        TN = confusion[0,0]
        FP = confusion[0,1]
        FN = confusion[1,0]
        print('True Positive: ', TP)
        print('True Negative: ', TN)
        print('False Positive: ', FP)
        print('False Negative: ', FN)
        print('Misclassifier Rate: ', (FP+FN)/float(TP+TN+FP+FN))
        print('Recall Rate: ', TP/float(TP + FN))
        print('Missing Rate: ', FN / float(TP + FN))
        print('False Positive Rate / False Alarm Rate: ', FP/float(TN+FP))
        print('True Negative Rate: ', TN/float(TN+FP))
        print('Precision: ', TP/float(TP+FP))
