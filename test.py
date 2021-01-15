# Natural Language Processing

# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import Sequential
from nltk.stem.porter import PorterStemmer
import pickle
import re

# %% Importing the dataset
dataset = pd.read_csv(r'Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
# parameter delimiter = '\t': file được đọc ngăn cách bởi dấu tab => đọc file tsv
# .tsv nghĩa là: Tab Separated Values
# quoting = 3: bỏ qua các kí tự "",'',...

# %% Cleaning the texts


# %% mở file chứa stopwords
stopwords = []
file = open('stopwords.txt', "r")
try:
    content = file.read()
    stopwords_list = content.split(",")
    for stopword in stopwords_list:
        s = stopword.replace('"', '')
        s = s.strip()
        stopwords.append(s)
finally:
    file.close()
# %%
    

count_stopword = []
stopword_list_appear_0 = []
non_stopword_list_appear_0 = []
stopword_list_appear_1 = []
non_stopword_list_appear_1 = []
stopword_list_appear = []
corpus = []  # chứa các reviews đã qua các bước lọc
for i in range(0, dataset.shape[0]):
    if(dataset['Liked'][i]==1):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # loại bỏ c    ác phần không phải
        # là chữ cái, thay thế bằng dấu spaces
        # ^: not
        # a-zA-Z: a to z nor A to Z
        # ' ': space
        review = review.lower()  # all to lowercase
        review = review.split()  # split to words
        ps = PorterStemmer()  # ran => run,....
        word_list = []
    
        for word in review:
            if word in set(stopwords):
                stopword_list_appear_1.append(word)
                # print(word)
            else:
                non_stopword_list_appear_1.append(word)
                word_list.append(ps.stem(word))
        review = ' '.join(word_list)  # nối lại các từ thành câu
        corpus.append(review)
    else:
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # loại bỏ c    ác phần không phải
        # là chữ cái, thay thế bằng dấu spaces
        # ^: not
        # a-zA-Z: a to z nor A to Z
        # ' ': space
        review = review.lower()  # all to lowercase
        review = review.split()  # split to words
        ps = PorterStemmer()  # ran => run,....
        word_list = []

        for word in review:
            if word in set(stopwords):
                stopword_list_appear_0.append(word)
                # print(word)
            else:
                non_stopword_list_appear_0.append(word)
                word_list.append(ps.stem(word))
        review = ' '.join(word_list)  # nối lại các từ thành câu
        corpus.append(review)
        
stopword_list_appear = stopword_list_appear_0 + stopword_list_appear_1
     
non_stopword_list_appear = non_stopword_list_appear_1 + non_stopword_list_appear_0


f = open(r'corpus\corpus.pickle', 'wb')
pickle.dump(corpus, f)
f.close()


f = open(r'runtime\stopword_at_runtime_for_bad_review.txt', "w+")
content = '\n'.join(stopword_list_appear_0)
f.write(content)
f.close()
f = open(r'runtime\stopword_at_runtime_for_good_review.txt', "w+")
content = '\n'.join(stopword_list_appear_1)
f.write(content)
f.close()

f2 = open(r'runtime\non_stopword_at_runtime_good_review.txt', "w+")
content = '\n'.join(non_stopword_list_appear_1)
f2.write(content)
f2.close()
f2 = open(r'runtime\non_stopword_at_runtime_bad_review.txt', "w+")
content = '\n'.join(non_stopword_list_appear_0)
f2.write(content)
f2.close()
f2 = open(r'runtime\non_stopword_at_runtime.txt', "w+")
content = '\n'.join(non_stopword_list_appear)
f2.write(content)
f2.close()
f2 = open(r'runtime\stopword_at_runtime.txt', "w+")
content = '\n'.join(stopword_list_appear_0)
f2.write(content)
f2.close()

# %% Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
# stored
mydictionary = sorted(cv.vocabulary_)
file = open(r'word_collection\words_in_BoW_model.txt', 'w+')
for key in mydictionary:
    file.write(key + "\n")
file.close()

f = open(r'vectorizer\cv.pickle', 'wb')
pickle.dump(cv, f)
f.close()

# %% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# %%

def preprocess_review_input(review):
    new_review = review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    new_review = [ps.stem(word) for word in new_review if not word in set(stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    return new_X_test


# %% ANN model
from keras import layers

model = Sequential()
input_dim = X_train.shape[1]  # input đầu vào bằng với số từ của tập dữ liệu
model = Sequential()
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # đầu ra là giá trị từ 0 đến 1

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=1)
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy1))
print("Testing Accuracy:  {:.4f}".format(accuracy2))
model.save('model')


# %% print
# def weight_of_model(model):
#     weights = []
#     for layer in model.layers:
#         weights.append(np.array(layer.get_weights()))
#     return np.array(weights)
# w = weight_of_model(model)
# %%
# y_pred=model.predict(review_input('If I could I would’ve given it zero stars. I opened my veggie bowl and found a giant piece of hardened avocado which had gone bad. Sometimes this location does amazing, sometimes horrible.'))

def review_input(review):
    y_pred = model.predict(preprocess_review_input(review))
    print(y_pred)
    return y_pred[0][0]


# %% Thống kê











#%% top 10 từ khoong trong tập stopword được gặp nhiều nhất (goood)

# danh sách các từ nằm trong stopword được xuất hiện trong tập dữ liệu cho ket qua 1
dict_non_stw_1 = {word : non_stopword_list_appear_1.count(word) for word in non_stopword_list_appear_1}
dict_non_stw_1 = {k : v for k, v in sorted(dict_non_stw_1.items(), key  = lambda item:item[1])}


list_non_stw_top10_1 = dict_non_stw_1.items()
list_non_stw_top10_1 = list(list_non_stw_top10_1)[-10:]

Xline = []
yline = []
for x in list_non_stw_top10_1:
    Xline.append(x[0])
    yline.append(x[1])
Xline.reverse()
yline.reverse()

plot1 = plt.figure(7)
plt.bar(Xline, yline)
plt.title('Top 10 non stopwords in good review')
plt.xlabel('word')
plt.ylabel('count')
plt.show()


#%% top 10 từ khoong trong tập stopword được gặp nhiều nhất (bad)

# danh sách các từ nằm trong stopword được xuất hiện trong tập dữ liệu cho ket qua 0
dict_non_stw_0 = {word : non_stopword_list_appear_0.count(word) for word in non_stopword_list_appear_0}
dict_non_stw_0 = {k : v for k, v in sorted(dict_non_stw_0.items(), key  = lambda item:item[1])}
# chonj ra top 10
list_non_stw_top10_0 = dict_non_stw_0.items()
list_non_stw_top10_0 = list(list_non_stw_top10_0)[-10:]

Xline = []
yline = []
for x in list_non_stw_top10_0:
    Xline.append(x[0])
    yline.append(x[1])
Xline.reverse()
yline.reverse()

plot1 = plt.figure(6)
plt.bar(Xline, yline)
plt.title('Top 10 non stopwords in bad review')
plt.xlabel('word')
plt.ylabel('count')
plt.show()



#%% top 10 từ trong tập stopword được gặp nhiều nhất

# danh sách các từ nằm trong stopword được xuất hiện trong tập dữ liệu
dict_stw = {word: stopword_list_appear.count(word) for word in stopword_list_appear}
dict_stw = {k: v for k, v in sorted(dict_stw.items(), key=lambda item: item[1])}
# chọn ra top 10 từ được gặp nhiều nhất
list_stw_top10 = dict_stw.items()
list_stw_top10 = list(list_stw_top10)[-10:]
Xline = []
yline = []
for x in list_stw_top10:
    Xline.append(x[0])
    yline.append(x[1])
Xline.reverse()
yline.reverse()

plot1 = plt.figure(1)
plt.bar(Xline, yline)
plt.title('Top 10 Stopwords')
plt.xlabel('stopword')
plt.ylabel('count')
plt.show()

#%% top 10 từ không có trong tập stopword được gặp nhiều nhất
# danh sách các từ không nằm trong stopword được xuất hiện trong tập dữ liệu        
dict_non_stw = {word: non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}
# chọn ra top 10 từ được gặp nhiều nhất
list_non_stw_top10 = dict_non_stw.items()
list_non_stw_top10 = list(list_non_stw_top10)[-10:]


Xline = []
yline = []
for x in list_non_stw_top10:
    Xline.append(x[0])
    yline.append(x[1])
Xline.reverse()
yline.reverse()

plot4 = plt.figure(4)
plt.bar(Xline, yline)
plt.title('Top 10 non-stopwords')
plt.xlabel('non-stopword')
plt.ylabel('count')
plt.show()

#%% so sánh số lượng stopword và không phải stopword
not_stopwords = len(sorted(cv.vocabulary_))
num_stopwords = len(dict_stw)
X_values = []
X_values.append(not_stopwords / (not_stopwords + num_stopwords))
X_values.append(num_stopwords / (not_stopwords + num_stopwords))
labe = ['non-stopword: ' + str(not_stopwords), 'stopwords: ' + str(num_stopwords)]

plot3 = plt.figure(3)
plt.pie(X_values, [0.1, 0], labels=labe)
plt.title('Stopword and non-stopword in database')
plt.show()

# so sánh số lần gặp các từ trong stopword và không trong stopword
not_stopwords_appear = len(non_stopword_list_appear)
num_stopwords_appear = len(stopword_list_appear)
X_values = []
X_values.append(not_stopwords_appear / (not_stopwords_appear + num_stopwords_appear))
X_values.append(num_stopwords_appear / (not_stopwords_appear + num_stopwords_appear))
labe = ['non-stopword: ' + str(not_stopwords_appear), 'stopwords: ' + str(num_stopwords_appear)]

plot5 = plt.figure(5)
plt.pie(X_values, [0.1, 0], labels=labe)
plt.title('Total stopword appear times and Total non-stopword appear times ')
plt.show()

#%% đánh giá hệ thống học máy

plot2 = plt.figure(2)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# plt2 = plt()
plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.ylim([0, 1])
plt.title('')
plt.legend()
plt.show()
