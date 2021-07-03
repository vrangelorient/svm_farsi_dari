#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import json
import re
from pprint import pprint
import matplotlib.pyplot as plt

import random; random.seed(44)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics


train = pd.read_json("news_subsample.json")
X_train, X_test, y_train, y_test = train_test_split(train["text"], train["language"], random_state=44, test_size=0.3)


tfidf = TfidfVectorizer(min_df=0.05, max_df=0.9)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)


svc = LinearSVC()
svc.fit(tfidf_train, y_train)
svc_pred = svc.predict(tfidf_test)
svc_score = metrics.accuracy_score(y_test, svc_pred)
print("LunearSVC score: ", svc_score)
cm = metrics.plot_confusion_matrix(svc, tfidf_test, y_test,display_labels=['Farsi', 'Dari'], cmap=plt.cm.Blues)
cm.ax_.set_title('Farsi/Dari LinearSVC Confusion Matrix')
plt.savefig("myplot.png")
plt.show()


import itertools

def plot_and_return_top_features(classifier, vectorizer, top_features=20):

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:top_features]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-top_features:]
    top_coefficients = np.hstack([topn_class1, topn_class2])
    if set(topn_class1).union(topn_class2):
        top_coefficients = topn_class1
        for ce in topn_class2:
            if ce not in topn_class1:
                top_coefficients.append(ce)

    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in [tc[0] for tc in top_coefficients]]
    plt.bar(np.arange(len(top_coefficients)), [tc[0] for tc in top_coefficients], color=colors)
    plt.xticks(np.arange(len(top_coefficients)),
               [tc[1] for tc in top_coefficients], rotation=60, ha='right')
    plt.show()
    return top_coefficients


top_features = plot_and_return_top_features(svc, tfidf)




