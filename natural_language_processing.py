# Natural Language Processing

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import Sequential
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # bỏ qua các từ không có nghĩa cho việc dự đoán
                                    # như: the, a, an, for, with, about,...
                                    
from nltk.stem.porter import PorterStemmer  #gôp các động từ đã được chia thì thành một
# ví dụ: run, ran running sẽ được tính chung là run để giảm kích thước dữ liệu

from Preprocessing import read_dataset, text_preprocessing

#%% Importing the dataset
dataset = read_dataset('Restaurant_Reviews.tsv')
# parameter delimiter = '\t': file được đọc ngăn cách bởi dấu tab => đọc file tsv
# .tsv nghĩa là: Tab Separated Values
# quoting = 3: bỏ qua các kí tự "",'',...

#%% Cleaning the texts
corpus = [] #chứa các reviews đã qua các bước lọc
corpus = text_preprocessing(dataset)


#%% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1600)
# cv = CountVectorizer(max_features = 1600)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

#%% Training the Naive Bayes model on the Training set


# from sklearn.svm import SVC
# classifier = SVC(kernel = 'linear', random_state = 0)
# classifier.fit(X_train, y_train)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)





# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

#%%Predicting the Test set results
y_pred = classifier.predict(X_test)


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



#%% CNN model
# model = Sequential()


#%% Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))



#%% test
def review_input(review):
    new_review = review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    # print(new_review)
    new_review = new_review.lower()
    # print(new_review)
    new_review = new_review.split()
    # print(new_review)
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove("isn't")
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    # print(new_review)
    new_review = ' '.join(new_review)
    # print(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    return new_y_pred[0]



