#EDA Packages
import pandas as pd
import numpy as np

# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset from https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection#
df1 = pd.read_csv("Youtube01-Psy.csv")

df1.head()

# Load all our dataset to merge them
df2 = pd.read_csv("Youtube02-KatyPerry.csv")
df3 = pd.read_csv("Youtube03-LMFAO.csv")
df4 = pd.read_csv("Youtube04-Eminem.csv")
df5 = pd.read_csv("Youtube05-Shakira.csv")

frames = [df1,df2,df3,df4,df5]

# Merging or Concatenating our DF
df_merged = pd.concat(frames)

# Total Size
df_merged.shape
(1956, 5)

# Merging with Keys
keys = ["Psy","KatyPerry","LMFAO","Eminem","Shakira"]

df_with_keys = pd.concat(frames,keys=keys)

df_with_keys

df_with_keys.loc['Shakira']

# Save and Write Merged Data to csv
df_with_keys.to_csv("YoutubeSpamMergeddata.csv")

df = df_with_keys

df.size
df.columns
df.dtypes
df.isnull().isnull().sum()
# Checking for Date
df["DATE"]
df.AUTHOR

df_data = df[["CONTENT","CLASS"]]

df_data.columns
df_x = df_data['CONTENT']
df_y = df_data['CLASS']


cv = CountVectorizer()
ex = cv.fit_transform(["Great song but check this out","What is this song?"])
[26]
ex.toarray()
cv.get_feature_names()
# Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus) # Fit the Data
X.toarray()
# get the feature names
cv.get_feature_names()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
X_train

# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

## Predicting with our model
clf.predict(X_test)

# Sample Prediciton
comment = ["Check this out"]
vect = cv.transform(comment).toarray()

clf.predict(vect)
class_dict = {'ham':0,'spam':1}
class_dict.values()

if clf.predict(vect) == 1:
    print("Spam")
else:
    print("Ham")

# Sample Prediciton 2
comment1 = ["Great song Friend"]
vect = cv.transform(comment1).toarray()
clf.predict(vect)

import pickle

naivebayesML = open("YtbSpam_model.pkl","wb")
pickle.dump(clf,naivebayesML)
naivebayesML.close()

# Load the model

ytb_model = open("YtbSpam_model.pkl","rb")
new_model = pickle.load(ytb_model)
new_model

# Sample Prediciton 3
comment2 = ["Hey Music Fans I really appreciate all of you,but see this song too"]
vect = cv.transform(comment2).toarray()
new_model.predict(vect)

if new_model.predict(vect) == 1:
    print("Spam")
else:
    print("Ham")
