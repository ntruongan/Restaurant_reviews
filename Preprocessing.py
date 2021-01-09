
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
from sklearn.feature_extraction.text import CountVectorizer                       
from nltk.stem.porter import PorterStemmer  #gôp các động từ đã được chia thì thành một
# ví dụ: run, ran running sẽ được tính chung là run để giảm kích thước dữ liệu



        
        

#%% Importing the dataset
def read_dataset(tsv_filename):
    """Return dataset for file tsv
    Parameter: tsv_filename: tsv file name
    """
    dataset = pd.read_csv(tsv_filename, delimiter = '\t', quoting = 3)
    return dataset
# parameter delimiter = '\t': file được đọc ngăn cách bởi dấu tab => đọc file tsv
# .tsv nghĩa là: Tab Separated Values
# quoting = 3: bỏ qua các kí tự "",'',...

#%% Cleaning the texts

def text_preprocessing(dataset):
    """return list of reviews after clear"""
    corpus = [] #chứa các reviews đã qua các bước lọc
    for i in range(0, dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        # review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #loại bỏ các phần không phải
        #là chữ cái, thay thế bằng dấu spaces
        #^: not
        #a-zA-Z: a to z nor A to Z
        #' ': space
        review = review.lower() # all to lowercase
        review = review.split() # split to words
        # print (review)
        ps = PorterStemmer() # ran => run,....
      
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')

        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    
        review = ' '.join(review) # nối lại các từ thành câu
        # print(review)
        corpus.append(review)
    return corpus
    


#%%

#%%

def text_for_nn_predict(review):
    new_review = review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray() # tao vector
    return new_X_test


