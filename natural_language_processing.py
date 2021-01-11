# Natural Language Processing

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import Sequential
import re
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords # bỏ qua các từ không có nghĩa cho việc dự đoán
                                    # như: the, a, an, for, with, about,...
                                    
from nltk.stem.porter import PorterStemmer  #gôp các động từ đã được chia thì thành một
# ví dụ: run, ran running sẽ được tính chung là run để giảm kích thước dữ liệu

# from Preprocessing import read_dataset, text_preprocessing

#%% Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# parameter delimiter = '\t': file được đọc ngăn cách bởi dấu tab => đọc file tsv
# .tsv nghĩa là: Tab Separated Values
# quoting = 3: bỏ qua các kí tự "",'',...
stopwords = []
file = open('stopwords.txt', "r")
try:
    content = file.read()
    stopwords_list = content.split(",")
    for stopword in stopwords_list:
        s = stopword.replace('"','')
        s = s.strip()
        stopwords.append(s)
finally:
    file.close()

#%% Cleaning the texts
count_stopword = []
stopword_list_appear = []
non_stopword_list_appear = []
corpus = [] #chứa các reviews đã qua các bước lọc
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #loại bỏ các phần không phải
        #là chữ cái, thay thế bằng dấu spaces
        #^: not
        #a-zA-Z: a to z nor A to Z
        #' ': space
    review = review.lower() # all to lowercase
    review = review.split() # split to words
    ps = PorterStemmer() # ran => run,....
    word_list = []
    
    for word in review:
        
        if word in set(stopwords):
            stopword_list_appear.append(word)
            # print(word)
        else:
            non_stopword_list_appear.append(word)
            word_list.append(ps.stem(word))

    # review = [ps.stem(word) for word in review if not word in set(stopwords)] # chuyển về nguyên
    # mẫu các từ không có trong stopwords
    # print(word_list)
    review = ' '.join(word_list) # nối lại các từ thành câu
    corpus.append(review)    
    
f = open(r'runtime\stopword_at_runtime.txt',"w+")
content = '\n'.join(stopword_list_appear)
f.write(content)
f.close()

f2 = open(r'runtime\non_stopword_at_runtime.txt',"w+")
content = '\n'.join(non_stopword_list_appear)
f2.write(content)
f2.close()


#%% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
# cv = CountVectorizer(max_features = 1600)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#%% Training the Naive Bayes model on the Training set

import pickle
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
f = open(r'classifier\svc_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
f = open(r'classifier\multinomialNB_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()



from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
f = open(r'classifier\gaussianNB_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()


from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
f = open(r'classifier\bernoulliNB_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

# import pickle
# f = open('my_classifier.pickle', 'rb')
# classifier_opened = pickle.load(f)
# f.close()

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



