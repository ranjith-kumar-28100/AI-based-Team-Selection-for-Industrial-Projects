import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class UnsupervisedClusterring:
  def __init__(self,data):
    self.data = data
    self.data_new = data
  
  def set_data(self,data):
    self.data= data

  def unsupervised_cluster(self,n):
    cluster = KMeans(n_clusters= n)
    y = cluster.fit_predict(self.data)
    self.data['Cluster'] = y
    return y

  def apply_rules(self,n,columns,weights):
    Count = []
    for i in range(0,n):
      avg_ = 0
      for j in range(len(columns)):
        avg_ += data_new[data_new['Cluster']==i][columns[j]].mean()*weights[j]
      Count.append(avg_)
    return Count, Count.index(max(Count))
  
  def use_cluster(self,n):
    self.data_new = self.data[self.data['Cluster']==n]
