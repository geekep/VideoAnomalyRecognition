import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class myknn():

    def __init__(self, X_train=[], X_test=[], y_train=[], y_test=[]):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_knn(self, k=5):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(self.X_train, self.y_train)
        print(clf.score(self.X_train, self.y_train))
        self.clf = clf
        return clf

    def predict(self, clf):
        y_predict = clf.predict(pd.DataFrame(self.X_test))
        print(clf.score(self.X_test, self.y_test))

        y_predict_proba = clf.predict_proba(self.X_test)
        print(y_predict_proba[:5])
        np.argmax(y_predict_proba, axis=1)[:5]
        self.y_predict = y_predict
        return y_predict

    def plot_confusion_matrix(self, classes, normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, path="graphs"):
        """
        input：
            classes,
            title,
            cmap: color of map could be setup (https://matplotlib.org/examples/color/colormaps_reference.html),
            path: the path of saved confusion matrix
        """

        cm = confusion_matrix(self.y_test, self.y_predict)
        print("confusion metrix:\n", cm)

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
