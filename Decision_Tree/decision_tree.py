# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:58:45 2018

@author: Priyank
"""
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pydotplus
class DecisionTree:
    def __init__(self):
        self.clf=DecisionTreeClassifier(criterion='entropy')
    def load_dataset(self,file_name=""):
        raw_data=open(file_name,'rt')
        reader=csv.reader(raw_data,delimiter=',',quoting=csv.QUOTE_ALL)
        self.x=list(reader)
        self.my_data=np.asarray(self.x)
        return self.my_data
    def preprocess_data(self,raw_data):
        le=LabelEncoder()
        return le.fit_transform(raw_data.ravel()).reshape(*raw_data.shape)
    def train_model(self,x_pre, y_pre):
        self.x_train, self.x_test, self.y_train, self.y_test=train_test_split(x_pre, y_pre, random_state = 47, test_size = 0.25)
        self.clf.fit(self.x_train,self.y_train)
    def predict_val(self,x):
        y_pred=self.clf.predict(x)
        return y_pred
    def get_accuracy(self,x,y):
        return accuracy_score(y_true=y, y_pred=self.clf.predict(x))
    def draw_tree(self):
        y=['1','0']
        dot_data=tree.export_graphviz(self.clf,out_file=None, feature_names=self.my_data[0,:self.my_data.shape[1]-1], class_names=y)
        graph=pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf('tree.pdf')
def main():
    obj=DecisionTree()
    file_name="dataset/blood_data.csv"
    data=obj.load_dataset(file_name=file_name)
    y=data[1:,data.shape[1]-1:]
    x=data[1:,0:data.shape[1]-1]
    x=obj.preprocess_data(raw_data=x)
    obj.train_model(x_pre=x,y_pre=y)
    accuracy_test="%.2f" % (obj.get_accuracy(x=obj.x_test,y=obj.y_test)*100)
    accuracy_train="%.2f" % (obj.get_accuracy(x=obj.x_train,y=obj.y_train)*100)
    print(accuracy_test,'% accuracy on test data.')
    print(accuracy_train,'% accuracy on train data.')
    print('Predicted value : ',obj.predict_val(np.asarray([[2,50,12500,98]])))
    obj.draw_tree()
if __name__=='__main__':
    main()
    