import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class CollaborativeRecommender:
  def __init__(self,data):
    self.data = data
  def create_pivot(self,Index='Eid',Columns,Values):
    return self.data.pivot(index = Index,columns =Columns,values = Values).fillna(0)
  def create_sparse(self,piv):
    self.sparse_data = csr_matrix(data_pivot.values)
  def do_recommendation(self,ideal):
    model = NearestNeighbors(metric = 'cosine',algorithm ='brute')
    model.fit(self.sparse_data)
    ideal = ideal.reshape(1,-1)
    distances,indices = model.kneighbors(query,n_neighbors = 100)
    return distances,indices
  
    
