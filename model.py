# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 13:29:41 2021

@author: ntruo
"""
import keras
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
class Model():
       
    def __init__(self, model_path = 'model', is_ann = True, dataset_path = 'Restaurant_Reviews.tsv', corpus = [], stopword_name='stopwords.txt'):
        
        self.is_ann = is_ann
        if self.is_ann:
            self.model = keras.models.load_model(model_path)
        else:
            f = open(model_path, 'rb')
            self.model = pickle.load(f)
            f.close()
        self.dataset = pd.read_csv(dataset_path, delimiter = '\t', quoting = 3)
        
        stopwords = []
        file = open(stopword_name, "r",encoding="utf-8")
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
        
        # corpus = [] #chứa các reviews đã qua các bước lọc
        # for i in range(0, self.dataset.shape[0]):
        #     review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i]) #loại bỏ các phần không phải
        #         #là chữ cái, thay thế bằng dấu spaces
        #         #^: not
        #         #a-zA-Z: a to z nor A to Z
        #         #' ': space
        #     review = review.lower() # all to lowercase
        #     review = review.split() # split to words
        #     ps = PorterStemmer() # ran => run,....
        #     review = [ps.stem(word) for word in review if not word in set(self.stopwords)] # chuyển về nguyên
        #     # mẫu các từ không có trong stopwords
        #     review = ' '.join(review) # nối lại các từ thành câu
        #     corpus.append(review)   
        f = open(r'corpus\corpus.pickle', 'rb')
        self.corpus = pickle.load(f)
        f.close()
        
        f = open(r'vectorizer\cv.pickle', 'rb')
        self.cv = pickle.load(f)
        f.close()

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
        file = open(file_name, "r", encoding="utf-8")
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
        self.cv = CountVectorizer() # chọn ra 2000 từ
        self.cv.fit_transform(self.corpus).toarray()
        mydictionary = sorted(self.cv.vocabulary_)
        file = open(r'word_collection\words_in_BoW_model.txt', 'w+', encoding="utf-8")
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
        # print(y_pred)
        result = np.array(y_pred)
        return sum(result)
        
    def top_stopword_count(self, top: int):
        f = open(r"runtime\stopword_at_runtime.txt","r", encoding="utf-8")
        stopword_list_appear = f.read().splitlines()
        f.close()
        dict_stw = {word:stopword_list_appear.count(word) for word in stopword_list_appear}
        dict_stw = {k: v for k, v in sorted(dict_stw.items(), key=lambda item: item[1])}
        list_stw_top = dict_stw.items()
        list_stw_top = list(list_stw_top)[-top:]
        Xline = []
        yline = []
        for x in list_stw_top:
            Xline.append(x[0])
            yline.append(x[1])
        Xline.reverse()
        yline.reverse()
        plot1 = plt.figure(1)
        plt.bar(Xline, yline)
        plt.title('Top {0} Stopwords'.format(top))
        plt.xlabel('stopword')
        plt.ylabel('count')
        plt.show()
        
    def top_non_stopword_count(self, top: int):
        f = open(r"runtime\non_stopword_at_runtime.txt","r",encoding="utf-8")
        non_stopword_list_appear = f.read().splitlines()
        f.close()
        dict_non_stw = {word:non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
        dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}
        list_non_stw_top10 = dict_non_stw.items()
        list_non_stw_top10 = list(list_non_stw_top10)[-top:]
        Xline = []
        yline = []
        for x in list_non_stw_top10:
            Xline.append(x[0])
            yline.append(x[1])
        Xline.reverse()
        yline.reverse()  
        plot4 = plt.figure(4)
        plt.bar(Xline, yline)
        plt.title('Top {0} non stopwords'.format(top))
        plt.xlabel('non-stopword')
        plt.ylabel('count')
        plt.show()   
        
        
    def top_non_stopword_count_for_bad(self, top: int):
        f = open(r"runtime\non_stopword_at_runtime_bad_review.txt","r",encoding="utf-8")
        non_stopword_list_appear = f.read().splitlines()
        f.close()
        dict_non_stw = {word:non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
        dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}
        list_non_stw_top10 = dict_non_stw.items()
        list_non_stw_top10 = list(list_non_stw_top10)[-top:]
        Xline = []
        yline = []
        for x in list_non_stw_top10:
            Xline.append(x[0])
            yline.append(x[1])
        Xline.reverse()
        yline.reverse()  
        plot6 = plt.figure(6)
        plt.bar(Xline, yline)
        plt.title('Top {0} non stopwords for bad reviews'.format(top))
        plt.xlabel('non-stopword')
        plt.ylabel('count')
        plt.show()   
        
        
    def top_non_stopword_count_for_good(self, top: int):
        f = open(r"runtime\non_stopword_at_runtime_good_review.txt","r",encoding="utf-8")
        non_stopword_list_appear = f.read().splitlines()
        f.close()
        dict_non_stw = {word:non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
        dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}
        list_non_stw_top10 = dict_non_stw.items()
        list_non_stw_top10 = list(list_non_stw_top10)[-top:]
        Xline = []
        yline = []
        for x in list_non_stw_top10:
            Xline.append(x[0])
            yline.append(x[1])
        Xline.reverse()
        yline.reverse()  
        plot6 = plt.figure(6)
        plt.bar(Xline, yline)
        plt.title('Top {0} non stopwords for good reviews'.format(top))
        plt.xlabel('non-stopword')
        plt.ylabel('count')
        plt.show()  

    def stopword_and_non_stopword(self):
        f = open(r"runtime\stopword_at_runtime.txt","r",encoding="utf-8")
        stopword_list_appear = f.read().splitlines()
        f.close()
        f = open(r"runtime\non_stopword_at_runtime.txt","r",encoding="utf-8")
        non_stopword_list_appear = f.read().splitlines()
        f.close()
        dict_stw = {word:stopword_list_appear.count(word) for word in stopword_list_appear}
        dict_stw = {k: v for k, v in sorted(dict_stw.items(), key=lambda item: item[1])}       
        dict_non_stw = {word:non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
        dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}
        not_stopwords = len(sorted(self.cv.vocabulary_))
        num_stopwords = len(dict_stw)
        X_values = []
        X_values.append(not_stopwords/(not_stopwords+num_stopwords))
        X_values.append(num_stopwords/(not_stopwords+num_stopwords))
        labe = ['non-stopword: '+str(not_stopwords),'stopwords: '+str(num_stopwords)]
        
        plot3 = plt.figure(3)
        plt.pie(X_values,[0.1,0],labels=labe)
        plt.title('Stopword and non-stopword in database')
        plt.show()
    
    def total_stopword_appear_total_non_stopword_appear(self):
        f = open(r"runtime\stopword_at_runtime.txt","r",encoding="utf-8")
        stopword_list_appear = f.read().splitlines()
        f.close()
        f = open(r"runtime\non_stopword_at_runtime.txt","r",encoding="utf-8")
        non_stopword_list_appear = f.read().splitlines()
        f.close()
        dict_stw = {word:stopword_list_appear.count(word) for word in stopword_list_appear}
        dict_stw = {k: v for k, v in sorted(dict_stw.items(), key=lambda item: item[1])}       
        dict_non_stw = {word:non_stopword_list_appear.count(word) for word in non_stopword_list_appear}
        dict_non_stw = {k: v for k, v in sorted(dict_non_stw.items(), key=lambda item: item[1])}

        not_stopwords_appear = len(non_stopword_list_appear)
        num_stopwords_appear = len(stopword_list_appear)
        X_values = []
        X_values.append(not_stopwords_appear/(not_stopwords_appear+num_stopwords_appear))
        X_values.append(num_stopwords_appear/(not_stopwords_appear+num_stopwords_appear))
        labe = ['non-stopword: '+str(not_stopwords_appear),'stopwords: '+str(num_stopwords_appear)]
        
        plot5 = plt.figure(5)
        plt.pie(X_values,[0.1,0],labels=labe)
        plt.title('Total stopword appear times and Total non-stopword appear times ')
        plt.show()
        

            

            
        
        
        