import pandas as pd
import statistics
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self,data):
        self.data = data
        self.columns = data.columns
        self.size = data.shape[0]
        print("Suceessfully Created Instance for the class")

    def handle_missing(self):
        for col in self.columns:
            if (self.data[col].dtype == object )and (self.data[col].isnull().sum() >0):
                self.data[col] = self.data[col].fillna(statistics.mode(self.data[col]))
            elif (self.data[col].dtype != object )and (self.data[col].isnull().sum() >0):                                                        
                self.data[col] = self.data[col].fillna(self.data[col].mean())
        print("Suceessfully Removed Missing Values in the data")


    def encoding_one_hot(self):
        self.data = pd.get_dummies(self.data,drop_first = True)
        self.columns = self.data.columns
        print("Suceessfully Encoded Categorical Values in the data")

    def sampling(self,choice =False,size = 1000):
        if choice:
            self.data= self.data.sample(n= size)
            print("Sucessfully sampled the data")
    
    def remove(self,cols):
      self.data  =self.data.drop(cols,axis =1)
      print("Columns Removed Sucessfully")

    def scaling(self,choice = False):
        if choice:
            scaler = MinMaxScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.data))
            self.data.columns = self.columns
            print("Sucessfully scaled the data")

