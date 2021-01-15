# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:37:32 2021

@author: MDTus
"""

import SMS_Spam as main

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Creating Pypeline for Naive Bayes
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer = main.data_pricessing)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])


# Splitting Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(main.dataSet['messages'],
            main.dataSet['label'], test_size=0.33, random_state=42)


# Training Data in Naive Bayes
pipeline.fit(X_train,y_train)

# Prediction
prediction = pipeline.predict(X_test)

# Report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print('\n\n')
print("Confusion Matrix")
print(confusion_matrix(y_test,prediction))
print('\n\n')
print("Classificaton Report")
print(classification_report(y_test,prediction))
print('\n\n')
print("Accuracy : ",accuracy_score(y_test, prediction,'\n\n\n'))
#................................................
