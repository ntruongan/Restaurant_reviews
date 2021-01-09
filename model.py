# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:29:41 2021

@author: ntruo
"""
import keras
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer

class Model():
       
    def __init__(self, model_path = 'model', dataset_path = 'Restaurant_Reviews.tsv', corpus = [], stopword_name='stopwords.txt'):
        self.model = keras.models.load_model(model_path)
        self.dataset = pd.read_csv(dataset_path, delimiter = '\t', quoting = 3)
        
        stopwords = []
        file = open(stopword_name, "r")
        try:
            content = file.read()
            stopwords_list = content.split(",")
            for stopword in stopwords_list:
                s = stopword.replace('"','')
                s = s.strip()
                stopwords.append(s)
        finally:
            file.close()
        self.stopwords = stopwords
        
        corpus = [] #chứa các reviews đã qua các bước lọc
        for i in range(0, self.dataset.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i]) #loại bỏ các phần không phải
                #là chữ cái, thay thế bằng dấu spaces
                #^: not
                #a-zA-Z: a to z nor A to Z
                #' ': space
            review = review.lower() # all to lowercase
            review = review.split() # split to words
            ps = PorterStemmer() # ran => run,....
            review = [ps.stem(word) for word in review if not word in set(self.stopwords)] # chuyển về nguyên
            # mẫu các từ không có trong stopwords
            review = ' '.join(review) # nối lại các từ thành câu
            corpus.append(review)       
        self.corpus = corpus   
        
        self.cv = CountVectorizer(max_features = 2000) # chọn ra 2000 từ
        self.cv.fit_transform(self.corpus).toarray()

    def get_dataset(self, loc_path = 'Restaurant_Reviews.tsv'):
        self.dataset = pd.read_csv(loc_path, delimiter = '\t', quoting = 3)
        
    def get_model(self, location_path = 'model'):
        self.model = keras.models.load_model(location_path)
        
    def get_corpus(self):
        corpus = [] #chứa các reviews đã qua các bước lọc
        for i in range(0, self.dataset.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i]) #loại bỏ các phần không phải
                #là chữ cái, thay thế bằng dấu spaces
                #^: not
                #a-zA-Z: a to z nor A to Z
                #' ': space
            review = review.lower() # all to lowercase
            review = review.split() # split to words
            ps = PorterStemmer() # ran => run,....
            review = [ps.stem(word) for word in review if not word in set(self.stopwords)] # chuyển về nguyên
            # mẫu các từ không có trong stopwords
            review = ' '.join(review) # nối lại các từ thành câu
            corpus.append(review)
        self.corpus = corpus
        
    def get_stopword_list(self, file_name = 'stopwords.txt'):
        stopwords = []
        file = open(file_name, "r")
        try:
            content = file.read()
            stopwords_list = content.split(",")
            for stopword in stopwords_list:
                s = stopword.replace('"','')
                s = s.strip()
                stopwords.append(s)
            self.stopwords = stopwords
        finally:
            file.close()
        
    def create_count_vector(self):
        self.cv = CountVectorizer(max_features = 2000) # chọn ra 2000 từ
        self.cv.fit_transform(self.corpus).toarray()
        mydictionary = sorted(self.cv.vocabulary_)
        file = open('words_in_BoW_model.txt', 'w+')
        for key in mydictionary:
            file.write(key+"\n")
        file.close()
        
    def preprocess_review_input(self, review):
        new_review = review
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        new_review = [ps.stem(word) for word in new_review if not word in set(self.stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = self.cv.transform(new_corpus).toarray()
        return new_X_test
    
    def review_input(self, review):
        y_pred=self.model.predict(self.preprocess_review_input(review))
        return y_pred[0][0]
        
        