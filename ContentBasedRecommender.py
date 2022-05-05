import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import sigmoid_kernel as sk
from scipy.sparse import csr_matrix
class ContentBasedRecommender:
  def __init__(self,data):
    self.data = data
  def create_matrix(self,Columns):
    df = self.data[Columns]
    self.df_matrix = csr_matrix(df.values)
  def create_sigmoid(self):
    self.sig = sk(self.df_matrix,self.df_matrix)
  def do_recommendation(self):
    indices = pd.Series(self.data.index,index = self.data.Eid)
    eid = self.data.iloc[0,'Eid']
    idx = indices[eid]
    sig_scores = list(enumerate(self.sig[idx]))
    sig_scores = sorted(sig_scores,key = lambda x:x[1],reverse = True)
    sig_scores = sig_scores[1:21]
    eid_indices = [i[0] for i in sig_scores]
    return self.data['Eid'].iloc[eid_indices]
    
