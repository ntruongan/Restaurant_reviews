# Natural Language Processing

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import Sequential
from Preprocessing import read_dataset, text_preprocessing

#%% Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# parameter delimiter = '\t': file được đọc ngăn cách bởi dấu tab => đọc file tsv
# .tsv nghĩa là: Tab Separated Values
# quoting = 3: bỏ qua các kí tự "",'',...

#%% Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # bỏ qua các từ không có nghĩa cho việc dự đoán
                                    # như: the, a, an, for, with, about,...
                                    
from nltk.stem.porter import PorterStemmer  #gôp các động từ đã được chia thì thành một
# ví dụ: run, ran running sẽ được tính chung là run để giảm kích thước dữ liệu
dataset = read_dataset('Restaurant_Reviews.tsv')
corpus = [] #chứa các reviews đã qua các bước lọc
corpus = text_preprocessing(dataset)


#%% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1600)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

#%% 

def preprocess_review_input(review):
    new_review = review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove("isn't")
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    return new_X_test

#%% ANN model
from keras import layers
model = Sequential()
input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',              
              optimizer='adam',               
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                     epochs=100,
                     verbose=1,#  will show you nothing (silent)
# verbose=1 will show you an animated progress bar :
                     validation_data=(X_test, y_test),
                     batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=1)
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy1))
print("Testing Accuracy:  {:.4f}".format(accuracy2))


#%% print 
def weight_of_model(model):
    weights = []
    for layer in model.layers:
        weights.append(np.array(layer.get_weights()))
    return np.array(weights)
w = weight_of_model(model)
#%%
# y_pred=model.predict(review_input('If I could I would’ve given it zero stars. I opened my veggie bowl and found a giant piece of hardened avocado which had gone bad. Sometimes this location does amazing, sometimes horrible.'))
def review_input(review):
    y_pred=model.predict(preprocess_review_input(review))
    print(y_pred)
    return y_pred[0][0]
#%% Making the Confusion Matrix

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print(accuracy_score(y_test, y_pred))



#%% test



