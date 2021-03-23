import nltk
import csv
import re
import pickle
import numpy
import pandas as pd
import numpy as np
import string
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Opening the tsv files and placing them into variables
drugLibTest_raw = open("C:\\Users\\Lance\\Downloads\\drugLib_raw\\drugLibTest_raw.tsv")
drugLibTrain_raw = open("C:\\Users\\Lance\\Downloads\\drugLib_raw\\drugLibTrain_raw.tsv")

#Put streams into pandas dataframe
drugTestFrame = pd.read_csv(drugLibTest_raw, sep='\t')
drugTrainFrame = pd.read_csv(drugLibTrain_raw, sep='\t')

#combining the two dataframes together
drugList = pd.concat([drugTestFrame, drugTrainFrame])

#drop the column Unnamed
drugList = drugList.drop(['Unnamed: 0'], axis=1)

#drop null values
drugList = drugList.dropna()

drugList = drugList[['rating', 'urlDrugName', 'effectiveness', 'sideEffects', 'condition', 'benefitsReview', 'sideEffectsReview', 'commentsReview']]

#checking if there are null values in the data
# print("THERE ARE THIS AMOUNT OF NULL VALUES: ")
# print(drugList.isna().sum())

drugList['combinedReview'] = drugList[['benefitsReview', 'sideEffectsReview', 'commentsReview']].agg(' '.join, axis=1)

#PREPROCESSING
#Replace commas with space
drugList['combinedReview'] = drugList['combinedReview'].str.replace(',', ' ')

# Remove all the special characters
drugList['combinedReview'] = drugList['combinedReview'].str.replace(r'\W', ' ')

# Removing prefixed 'b'
drugList['combinedReview'] = drugList['combinedReview'].str.replace(r'^b\s+', ' ')

# Converting to Lowercase
drugList['combinedReview'] = drugList['combinedReview'].str.lower()

# remove numbers
drugList['combinedReview'] = drugList['combinedReview'].str.replace(r' \d+', ' ')

# Substituting multiple spaces with single space
drugList['combinedReview'] = drugList['combinedReview'].str.replace(r'\s+', ' ')

# remove all single characters
drugList['combinedReview'] = drugList['combinedReview'].str.replace(r'\s+[a-zA-Z]\s+', ' ')

#Stemming words
stemmer = WordNetLemmatizer()
drugList['combinedReview'] = [stemmer.lemmatize(word) for word in drugList['combinedReview']]

# Function for extracting ONLY NOUNS AND VERBS
# # Getting nouns and verbs
# def get_adjectives(text):
#     blob = TextBlob(text)
#     return [word for (word, tag) in blob.tags if tag == 'NN' or tag == 'VB']
#
# drugList['combinedReview'] = drugList['combinedReview'].apply(get_adjectives)
#
# # Function for converting LIST TO STRINGS
# # Reducing List to Strings
# def listToString(s):
#     str1 = ""
#
#     for ele in s:
#         str1 += " " + ele
#
#     return str1
#
# drugList['combinedReview'] = drugList['combinedReview'].apply(listToString)

#Label Encoding y values (Side Effects)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(drugList['sideEffects'])

#Encoding the X features
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(drugList[['combinedReview']])

stop_words = set(stopwords.words('english'))

#For sentences
from sklearn.feature_extraction.text import TfidfVectorizer
tfidvectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
tfidvectorizer.fit(drugList['combinedReview'])
X = tfidvectorizer.fit_transform(drugList['combinedReview'])

# #Test Splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("RandomForest Classifier Accuracy:", accuracy_score(y_test, y_pred))

# THIS IS THE FPR VALUES and report
#######################################
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
######################################

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Bernoulli NB Accuracy: ", accuracy_score(y_test, y_pred))

# THIS IS THE FPR VALUES and report
#######################################
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#######################################

#Uncomment these lines below if you want to print the final csv file out
# drugList.sort_values(['rating'], ascending=False, inplace=True)
# drugList.to_csv('drugLib_processed3.csv', index=False)

# import os
# os.stat('drugLib_processed3.csv').st_size
# print(drugList.info())