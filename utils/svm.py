import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools


def save_model(clf, src):
    joblib.dump(clf, src)
    print("model has been saved to" + src)


def get_model(src):
    model = joblib.load(src)
    print("model has been loaded" + src)
    return model


class mysvm():

    def __init__(self, X_train=[], X_test=[], y_train=[], y_test=[]):
        # preprocessing
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = label_binarize(y_train, classes=range(y_train.length))
        self.y_test = label_binarize(y_test, classes=range(y_test.length))

    def train_svm(self):
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape='ovo', degree=3, gamma=5,
                      kernel='rbf', max_iter=-1, probability=True,
                      random_state=None, shrinking=True, tol=0.001,
                      verbose=False)
        clf.fit(self.X_train, self.y_train)
        print(clf.score(self.X_train, self.y_train))
        return clf

    def predit(self, clf):
        y_predict = clf.predict(pd.DataFrame(self.X_test))
        print(clf.score(self.X_test, self.y_test))

        y_predict_scores = clf.decision_function(self.X_test)
        print(y_predict_scores[:5])
        np.argmax(clf.decision_function(self.X_test), axis=1)[:5]

        y_predict_proba = clf.predict_proba(self.X_test)
        print(y_predict_proba[:5])
        np.argmax(y_predict_proba, axis=1)[:5]

        acc_overall = accuracy_score(y_predict, self.y_test)
        print("overall accuracy: %f" % acc_overall)

        acc_for_each_class = precision_score(self.y_test, y_predict, average=None)
        print("accuracy for each class:\n", acc_for_each_class)

        acc_avg = np.mean(acc_for_each_class)
        print("average accuracy: %f" % acc_avg)
        classification_rep = classification_report(self.y_test, y_predict,
                                                   target_names=self.y_train)
        print("classification report:\n", classification_rep)

        kappa = cohen_kappa_score(y_predict, self.y_test)
        print("kappa: %f" % kappa)

        cm = confusion_matrix(self.y_test.argmax(axis=1), y_predict.argmax(axis=1))
        print("confusion metrix:\n", cm)

        return y_predict

    def plot_confusion_matrix(self, cm, classes, normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, path="graphs"):
        """
        input：
            cm: confusion matrix,
            classes,
            title,
            cmap: color of map could be setup (https://matplotlib.org/examples/color/colormaps_reference.html),
            path: the path of saved confusion matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix without normalization')
        print(cm)

        plt.figure(figsize=(11, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.figure(facecolor='w')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.grid(True)
        plt.savefig(path, dpi=300)
        plt.show()
