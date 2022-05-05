from tkinter import *
from tkinter import ttk
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statistics
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import sigmoid_kernel as sk

class DataSet:
    def __init__(self, data):
        self.data = data
        self.columns = data.columns
        self.size = data.shape[0]
        print("Suceessfully Created Instance for the class")

    def handle_missing(self):
        for col in self.columns:
            if (self.data[col].dtype == object) and (self.data[col].isnull().sum() > 0):
                self.data[col] = self.data[col].fillna(
                    statistics.mode(self.data[col]))
            elif (self.data[col].dtype != object) and (self.data[col].isnull().sum() > 0):
                self.data[col] = self.data[col].fillna(self.data[col].mean())
        print("Suceessfully Removed Missing Values in the data")

    def encoding_one_hot(self):
        self.data = pd.get_dummies(self.data, drop_first=True)
        self.columns = self.data.columns
        print("Suceessfully Encoded Categorical Values in the data")

    def sampling(self, choice=False, size=1000):
        if choice:
            self.data = self.data.sample(n=size)
            print("Sucessfully sampled the data")

    def remove(self, cols):
        self.data = self.data.drop(cols, axis=1)
        print("Columns Removed Sucessfully")

    def scaling(self, choice=False):
        if choice:
            scaler = StandardScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.data))
            self.data.columns = self.columns
            print("Sucessfully scaled the data")


class UnsupervisedCluster:
    def __init__(self, data):
        self.data = data
        self.data_new = data

    def set_data(self, data):
        self.data = data

    def unsupervised_cluster(self, n):
        cluster = KMeans(n_clusters=n)
        y = cluster.fit_predict(self.data)
        self.data['Cluster'] = y
        return y

    def apply_rules(self, n, columns, weights):
        Count = []
        for i in range(0, n):
            avg_ = 0.0
            for j in range(len(columns)):
                avg_ += self.data_new[self.data_new['Cluster']
                                      == i][columns[j]].mean()*weights[j]
            Count.append(avg_)
        return Count, Count.index(max(Count))

    def use_cluster(self, n):
        self.data_new = self.data[self.data['Cluster'] == n]


class CollaborativeRecommender:
    def __init__(self, data):
        self.data = data

    def create_pivot(self):
        col = []
        val = []
        ind = 'Eid'
        for i in self.data.columns:
            if '_ID' in i:
                col.append(i)
            elif i == 'Eid':
                continue
            else:
                val.append(i)
        self.data_pivot = self.data.pivot(
            index=ind, columns=col, values=val).fillna(0)
        

    def create_datamatrix(self):
        self.data_matrix = csr_matrix(self.data_pivot.values)

    def Recommender(self):
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.data_matrix)
        
    def Recommend(self,query):
        distances, indices = self.model.kneighbors(query, n_neighbors=50)
        return indices

class ContentBasedRecommender:
    def __init__(self,data):
        self.data = data
    def create_matrix(self,Columns):
        Columns  = Columns+ ['Rating', 'Total_projects']
        df = self.data[Columns]
        self.df_matrix = csr_matrix(df.values)
    def create_sigmoid(self):
        self.sig = sk(self.df_matrix,self.df_matrix)     
    def do_recommendation(self,eid):
        indices = pd.Series(self.data.index,index = self.data.Eid)
        idx = indices[eid]
        sig_scores = list(enumerate(self.sig[idx]))
        sig_scores = sorted(sig_scores,key = lambda x:x[1],reverse = True)
        sig_scores = sig_scores[1:21]
        eid_indices = [i[0] for i in sig_scores]
        return self.data['Eid'].iloc[eid_indices]

class DesignationMatching:
    def __init__(self,data):
        self.data = data
    def actual_des(self):
        print(Counter(self.data['Designation']))
    def assign_priority(self):
        counts=[0 for i in range(len(self.data))]
        for i in self.data.Designation.unique():
            count = 0
            for j in range(len(self.data)):
                if(self.data.iloc[j]['Designation'] == i):
                    count = count+1
                    counts[j] = count
        self.data['Priority']  = counts
    def designation_matching(self):
        str_=''''''
        for i in self.data.Priority.unique():
            sub = self.data[self.data.Priority == i]
            str_ = str_+'Priority--'+str(i)+'\n'
            for j in range(len(sub)):
                str_=str_+(str(sub.iloc[j]['Ename'])+" "+str(sub.iloc[j]['Designation']))+"\n"
            str_=str_+'--------------------------------------------------------------\n'
        return str_

class FirstPage:
  def __init__(self,master):
    self.window = master
    self.window.title("Enter the project details")
    self.window.geometry('1000x1000')
    self.window.configure(background = "tomato2")
    self.title= ttk.Label(self.window ,text = " Basic Details !!!!",width =20,background = "tomato2",anchor="e", justify=LEFT,font=("Bernard MT Condensed", 25)).grid(row = 0,column = 0)
    self.pname= ttk.Label(self.window ,text = " Project Name",width =20,background = "tomato2",anchor="e", justify=LEFT,font=("Copperplate Gothic Light", 15)).grid(row = 1,column = 0)
    self.cname = ttk.Label(self.window ,text = "Client Name",width =20,background = "tomato2",anchor="e", justify=LEFT,font=("Copperplate Gothic Light", 15)).grid(row = 2,column = 0)
    self.dname = ttk.Label(self.window ,text = "Domain Name ",width=20,background = "tomato2",anchor="e", justify=LEFT,font=("Copperplate Gothic Light", 15)).grid(row = 3,column = 0)
    self.pname_field =Entry(self.window,width=40)
    self.pname_field.grid(row = 1,column = 1)
    self.cname_field =Entry(self.window,width =40)
    self.cname_field.grid(row = 2,column = 1)
    self.dname_field = Entry(self.window,width  =40)
    self.dname_field.grid(row = 3,column = 1)
    self.btn = ttk.Button(self.window ,text="Next",command=self.insert,width = 20).grid(row=4,column=3)

  def insert(self):
    if (self.pname_field.get()==""or self.cname_field.get()=="" or self.dname_field.get()==""):
      print("Enter all the details")
    else:
      fields =[self.pname_field.get(),self.cname_field.get(),self.dname_field.get()]
      with open(r'C:/Users/Matrix/Desktop/Final_year_ project/Project_details.csv','a') as f:
        writer= csv.writer(f)
        writer.writerow(fields)
      print(fields)
      self.window.destroy()
      master = Tk()
      app = SecondPage(master)
      master.mainloop()
      
      
      print("Sucess!!!")
class SecondPage:
  def __init__(self,master):
    self.window = master
    self.window.title("Enter the feature importance details")
    self.window.geometry('1200x1200')
    self.window.configure(background = "tomato2")
    self.cn = ttk.Label(self.window ,text = "Feature Importance details!!!",background = "tomato2",font=("Bernard MT Condensed", 25)).grid(row = 0,column = 0)
    self.cn = ttk.Label(self.window ,text = "No. of Cluster",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 1,column = 0)
    self.f1= ttk.Label(self.window ,text = "Feature 1",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 2,column = 0)
    self.f2 = ttk.Label(self.window ,text = "Feature 2",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 3,column = 0)
    self.f3 = ttk.Label(self.window ,text = "Feature 3 ",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 4,column = 0)
    self.s1= ttk.Label(self.window ,text = "Score 1",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 2,column = 2)
    self.s2 = ttk.Label(self.window ,text = "Score 2",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 3,column = 2)
    self.s3 = ttk.Label(self.window ,text = "Score 3",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 4,column = 2)
    self.cn_field =Entry(self.window)
    self.cn_field.grid(row = 1,column = 1)    
    self.f1_field =Entry(self.window,width = 40)
    self.f1_field.grid(row = 2,column = 1)
    self.f2_field =Entry(self.window,width =40)
    self.f2_field.grid(row = 3,column = 1)
    self.f3_field = Entry(self.window,width =40)
    self.f3_field.grid(row = 4,column = 1)
    self.s1_field =Entry(self.window)
    self.s1_field.grid(row = 2,column = 3)
    self.s2_field =Entry(self.window)
    self.s2_field.grid(row = 3,column = 3)
    self.s3_field = Entry(self.window)
    self.s3_field.grid(row = 4,column = 3)
    self.l = ttk.Label(self.window, text="Available Options are given below",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row= 5,column=0)
    self.l1 = ttk.Label(self.window ,text = "Experience||Total_project||Rating",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 6,column = 0)
    self.l2 = ttk.Label(self.window ,text = "AI_project_count||ML_project_count||JS_project_count",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 7,column = 0)
    self.l3 = ttk.Label(self.window ,text = "Java_project_count||DotNet_project_count||Mobile_project_count",background = "tomato2",font=("Copperplate Gothic Light", 10)).grid(row = 8,column = 0)    
    self.btn = ttk.Button(self.window ,text="Next",command=self.insert,width =20).grid(row=9,column=2)

  def insert(self):
    if (self.cn_field.get()==""or self.f1_field.get()=="" or self.f2_field.get()=="" or self.f3_field.get()=="" or self.s1_field.get()=="" or self.s2_field.get()=="" or self.s3_field.get()==""):
      print("Enter all the details")
    else:
      fields =[self.cn_field.get(),self.f1_field.get(),self.f2_field.get(),self.f3_field.get(),self.s1_field.get(),self.s2_field.get(),self.s3_field.get()]
      with open(r'C:/Users/Matrix/Desktop/Final_year_ project/cluster_details.csv','a') as f:
        writer= csv.writer(f)
        writer.writerow(fields)
      print(fields) 
      print("Sucess!!!")
      self.window.destroy()
      master = Tk()
      app = ThirdPage(master)
      master.mainloop()
    

class ThirdPage:
  def __init__(self,master):
    self.window = master
    self.window.title("Enter the skill-set details")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    self.tt = ttk.Label(self.window,text="Enter The Skills Required (Yes/No) !!!!",background = 'tomato2' ,font = ("Bernard MT Condensed", 20)).grid(row = 0,column= 1)
    self.python = ttk.Label(self.window,text="Python",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 1,column= 0)
    self.s1 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton", background = "light green",foreground = "red", font = ("Copperplate Gothic Light", 10, "bold"))
    self.r1 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s1,
                value = 1)
    self.r1.grid(row = 1,column  =1)
    self.s1.set(0)

    self.r1 = ttk.Radiobutton(self.window, text = "No", variable = self.s1,
                value = 0)
    self.r1.grid(row = 1,column  =2)
    self.s1.set(0)

    self.ml = ttk.Label(self.window,text="Machine Learning",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 2,column= 0)
    self.s2 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r2 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s2,
                value = 1)
    self.r2.grid(row = 2,column  =1)

    self.r2 = ttk.Radiobutton(self.window, text = "No", variable = self.s2,
                value = 0)
    self.r2.grid(row = 2,column  =2)
    self.s2.set(0)

    self.dl = ttk.Label(self.window,text="Deep Learning",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 3,column= 0)
    self.s3 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r3 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s3,
                value = 1)
    self.r3.grid(row = 3,column  =1)

    self.r3 = ttk.Radiobutton(self.window, text = "No", variable = self.s3,
                value = 0)
    self.r3.grid(row = 3,column  =2)
    self.s3.set(0)

    self.da = ttk.Label(self.window,text="Data Analysis",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 4,column= 0)
    self.s4 = StringVar(self.window)
    style = ttk.Style(self.window)
    
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r4 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s4,
                value = 1)
    self.r4.grid(row = 4,column  =1)

    self.r4 = ttk.Radiobutton(self.window, text = "No", variable = self.s4,
                value = 0)
    self.r4.grid(row = 4,column  =2)
    self.s4.set(0)

    self.asp = ttk.Label(self.window,text="ASP.Net",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 5,column= 0)
    self.s5 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r5 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s5,
                value = 1)
    self.r5.grid(row = 5,column  =1)

    self.r5 = ttk.Radiobutton(self.window, text = "No", variable = self.s5,
                value = 0)
    self.r5.grid(row = 5,column  =2)
    self.s5.set(0)

    self.ado = ttk.Label(self.window,text="ADO.Net",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 6,column= 0)
    self.s6 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r6 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s6,
                value = 1)
    self.r6.grid(row = 6,column  =1)

    self.r6 = ttk.Radiobutton(self.window, text = "No", variable = self.s6,
                value = 0)
    self.r6.grid(row = 6,column  =2)
    self.s6.set(0)

    self.vb = ttk.Label(self.window,text="VB.Net",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 7,column= 0)
    self.s7 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r7 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s7,
                value = 1)
    self.r7.grid(row = 7,column  =1)

    self.r7 = ttk.Radiobutton(self.window, text = "No", variable = self.s7,
                value = 0)
    self.r7.grid(row = 7,column  =2)
    self.s7.set(0)

    self.csharp = ttk.Label(self.window,text="c#",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 8,column= 0)
    self.s8 = StringVar(self.window)
    style = ttk.Style(self.window)
    
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r8 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s8,
                value = 1)
    self.r8.grid(row = 8,column  =1)

    self.r8 = ttk.Radiobutton(self.window, text = "No", variable = self.s8,
                value = 0)
    self.r8.grid(row = 8,column  =2)
    self.s8.set(0)

    self.java = ttk.Label(self.window,text="Java",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 9,column= 0)
    self.s9 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r9 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s9,
                value = 1)
    self.r9.grid(row = 9,column  =1)

    self.r9 = ttk.Radiobutton(self.window, text = "No", variable = self.s9,
                value = 0)
    self.r9.grid(row = 9,column  =2)
    self.s9.set(0)

    self.sb = ttk.Label(self.window,text="Spring Boot",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 10  ,column= 0)
    self.s10 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r10 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s10,
                value = 1)
    self.r10.grid(row = 10,column  =1)

    self.r10 = ttk.Radiobutton(self.window, text = "No", variable = self.s10,
                value = 0)
    self.r10.grid(row = 10,column  =2)
    self.s10.set(0)

    self.hb = ttk.Label(self.window,text="Hibernate",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 11,column= 0)
    self.s11 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r11 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s11,
                value = 1)
    self.r11.grid(row = 11,column  =1)

    self.r11 = ttk.Radiobutton(self.window, text = "No", variable = self.s11,
                value = 0)
    self.r11.grid(row = 11,column  =2)
    self.s11.set(0)

    self.nlp = ttk.Label(self.window,text="NLP",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 12,column= 0)
    self.s12 = StringVar(self.window)
    style = ttk.Style(self.window)
    
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r12 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s12,
                value = 1)
    self.r12.grid(row = 12,column  =1)

    self.r12 = ttk.Radiobutton(self.window, text = "No", variable = self.s12,
                value = 0)
    self.r12.grid(row = 12,column  =2)
    self.s12.set(0)

    self.cv = ttk.Label(self.window,text="CV",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 13,column= 0)
    self.s13 = StringVar(self.window)
    style = ttk.Style(self.window)
    
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r13 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s13,
                value = 1)
    self.r13.grid(row = 13,column  =1)

    self.r13 = ttk.Radiobutton(self.window, text = "No", variable = self.s13,
                value = 0)
    self.r13.grid(row = 13,column  =2)
    self.s13.set(0)

    self.js = ttk.Label(self.window,text="JS",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 14,column= 0)
    self.s14 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r14 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s14,
                value = 1)
    self.r14.grid(row = 14,column  =1)

    self.r14 = ttk.Radiobutton(self.window, text = "No", variable = self.s14,
                value = 0)
    self.r14.grid(row = 14,column  =2)
    self.s14.set(0)

    self.react = ttk.Label(self.window,text="React",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 15  ,column= 0)
    self.s15 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r15 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s15,
                value = 1)
    self.r15.grid(row = 15,column  =1)

    self.r15 = ttk.Radiobutton(self.window, text = "No", variable = self.s15,
                value = 0)
    self.r15.grid(row = 15,column  =2)
    self.s15.set(0)

    self.node = ttk.Label(self.window,text="Node",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 16,column= 0)
    self.s16 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2', foreground = 'Light green',font = ("Copperplate Gothic Light", 10, "bold"))
    self.r16 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s16,
                value = 1)
    self.r16.grid(row = 16,column  =1)

    self.r16 = ttk.Radiobutton(self.window, text = "No", variable = self.s16,
                value = 0)
    self.r16.grid(row = 16,column  =2)
    self.s16.set(0)

    self.angular = ttk.Label(self.window,text="Angular",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 17,column= 0)
    self.s17 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r17 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s17,
                value = 1)
    self.r17.grid(row = 17,column  =1)

    self.r17 = ttk.Radiobutton(self.window, text = "No", variable = self.s17,
                value = 0)
    self.r17.grid(row = 17,column  =2)
    self.s17.set(0)

    self.dart = ttk.Label(self.window,text="Dart",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 18,column= 0)
    self.s18 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r18 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s18,
                value = 1)
    self.r18.grid(row = 18,column  =1)

    self.r18 = ttk.Radiobutton(self.window, text = "No", variable = self.s18,
                value = 0)
    self.r18.grid(row = 18,column  =2)
    self.s18.set(0)

    self.flutter = ttk.Label(self.window,text="Flutter",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(row = 19,column= 0)
    self.s19 = StringVar(self.window)
    style = ttk.Style(self.window)
    style.configure("TRadiobutton",background = 'tomato2',foreground = 'Light green', font = ("Copperplate Gothic Light", 10, "bold"))
    self.r19 = ttk.Radiobutton(self.window, text = "Yes", variable = self.s19,
                value = 1)
    self.r19.grid(row = 19,column  =1)

    self.r19 = ttk.Radiobutton(self.window, text = "No", variable = self.s19,
                value = 0)
    self.r19.grid(row = 19,column  =2)
    self.s19.set(0)

    self.btt = ttk.Button(self.window,text="Next",command= self.insert).grid(row = 20,column =2)


  def insert(self):
    if (self.s1.get()=='' or self.s2.get()=='' or self.s3.get()=='' or self.s4.get()=='' or self.s5.get()=='' or self.s6.get()=='' or self.s7.get()=='' or self.s8.get()=='' or self.s9.get()=='' or self.s10.get()=='' or self.s11.get()=='' or self.s12.get()=='' or self.s13.get()=='' or self.s14.get()=='' or self.s15.get()=='' or self.s16.get()=='' or self.s17.get()=='' or self.s18.get()=='' and self.s19.get()==''):
      print("Enter all the details")
    else:
      fields =[self.s1.get(),self.s2.get(),self.s3.get(),self.s4.get(),self.s5.get(),self.s6.get(),self.s7.get(),self.s8.get(),self.s9.get(),self.s10.get(),self.s11.get(),self.s12.get(),self.s13.get(),self.s14.get(),self.s15.get(),self.s16.get(),self.s17.get(),self.s18.get(),self.s19.get()]
      with open(r'C:/Users/Matrix/Desktop/Final_year_ project/skill_details.csv','a') as f:
        writer= csv.writer(f)
        writer.writerow(fields)
      print(fields) 
      print("Sucess!!!")
      self.window.destroy()
      master = Tk()
      app = FourthPage(master)
      master.mainloop()
    

class FourthPage:
  def __init__(self,master):
    self.window = master
    self.window.title("Enter the important content details")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    
    self.l1 = ttk.Label(self.window, text = "Important Content Details !!!!",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(row = 0, column = 1)
  
    self.l2 = ttk.Label(self.window, text = "Feature 1:",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 1, padx = 10, pady = 25)
      

    self.n1 = StringVar(self.window)
    self.monthchoosen = ttk.Combobox(self.window, width = 27, textvariable = self.n1)
      
    self.monthchoosen['values'] = ('AI_project_count','ML_project_count','JS_project_count','Java_project_count','DotNet_project_count','Mobile_project_count')
      
    self.monthchoosen.grid(column = 1, row = 1)
    self.monthchoosen.current()


      
    self.l2 = ttk.Label(self.window, text = "Feature 2:",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 2, padx = 10, pady = 25)
      

    self.n2 = StringVar(self.window)
    self.monthchoosen1 = ttk.Combobox(self.window, width = 27, textvariable = self.n2)
      
    self.monthchoosen1['values'] = ('Area_of_Interest_1_AI','Area_of_Interest_1_ML','Area_of_Interest_1_JS','Area_of_Interest_1_Java','Area_of_Interest_1_DotNet','Area_of_Interest_1_Mobile')
      
    self.monthchoosen1.grid(column = 1, row = 2)
    self.monthchoosen1.current()

    self.l3 = ttk.Label(self.window, text = "Feature 3:",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 3, padx = 10, pady = 25)
      

    self.n3 = StringVar(self.window)
    self.monthchoosen2 = ttk.Combobox(self.window, width = 27, textvariable = self.n3)
      
    self.monthchoosen2['values'] = ('Area_of_Interest_2_AI','Area_of_Interest_2_ML','Area_of_Interest_2_JS','Area_of_Interest_2_Java','Area_of_Interest_2_DotNet','Area_of_Interest_2_Mobile')
      
    self.monthchoosen2.grid(column = 1, row = 3)
    self.monthchoosen2.current()


    self.l4 = ttk.Label(self.window, text = "Feature 4:",background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 4, padx = 10, pady = 25)
      

    self.n4 = StringVar(self.window)
    self.monthchoosen3 = ttk.Combobox(self.window, width = 27,
                                 textvariable = self.n4)
      
    self.monthchoosen3['values'] = ('Area_of_Interest_3_AI','Area_of_Interest_3_ML','Area_of_Interest_3_JS','Area_of_Interest_3_Java','Area_of_Interest_3_DotNet','Area_of_Interest_3_Mobile')
      
    self.monthchoosen3.grid(column = 1, row = 4)
    self.monthchoosen3.current()


    self.b1 = ttk.Button(self.window,text="Next",command=self.insert).grid(row = 5, column = 1)

            
  def insert(self):
    if (self.n1.get()=="" or self.n2.get()=="" or self.n3.get()=="" or self.n4.get()=="") :
      print("Enter all the details")
      print([self.n1.get(),self.n2.get(),self.n3.get(),self.n4.get()])
    else:
      fields =[self.n1.get(),self.n2.get(),self.n3.get(),self.n4.get()]
      with open(r'C:/Users/Matrix/Desktop/Final_year_ project/final_details.csv','a') as f:
        writer= csv.writer(f)
        writer.writerow(fields)
      print(fields)
      self.window.destroy()
      master = Tk()
      app = FifthPage(master)
      master.mainloop()      
      print("Sucess!!!")
    
class FifthPage:
  def __init__(self,master):
    self.window = master
    self.window.title("Cluster Info.")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    self.l = ttk.Label(self.window, text = "Processing Wait",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)
    self.b1 = ttk.Button(self.window,text="calculate",command=self.calculate).grid(row = 600, column = 5)
    self.b2 = ttk.Button(self.window,text="Next",command=self.next).grid(row = 600, column = 10)
  def calculate(self):
    data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/final_data_for_clustering.csv')
    dataset = DataSet(data)
    c_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/cluster_details.csv')
    col = c_data.iloc[-1,:]
    cluster_model = UnsupervisedCluster(dataset.data)
    new_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/final_data_for_clustering_without_scaling.csv')
    y = cluster_model.unsupervised_cluster(col['cno'])
    new_data['Cluster'] = y
    cluster_model.set_data(new_data)  
    Count, cluster_index = cluster_model.apply_rules(int(col['cno']), [col['f1'],col['f2'],col['f3']], [float(col['s1']),float(col['s2']),float(col['s3'])])
    cluster_model.use_cluster(cluster_index)
    cluster_data = cluster_model.data_new
    index = cluster_data.index.values
    index = index+1
    self.l = ttk.Label(self.window, text = "                                         ",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)
    string_ = "There are "+str(col['cno'])+" clusters and cluster "+str(cluster_index)+" is selected"
    self.l1 = ttk.Label(self.window, text = string_,background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 1,
              row = 1)
    string_ = "The mean value for "+ str(col['f1'])+" is "+str(new_data[new_data['Cluster']==cluster_index][col['f1']].mean())
    self.l2 = ttk.Label(self.window, text = string_,background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 2)
    string_ = "The mean value for "+ str(col['f2'])+" is "+str(new_data[new_data['Cluster']==cluster_index][col['f2']].mean())
    self.l3 = ttk.Label(self.window, text = string_,background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 3)
    string_ = "The mean value for "+ str(col['f3'])+" is "+str(new_data[new_data['Cluster']==cluster_index][col['f3']].mean())
    self.l4 = ttk.Label(self.window, text = string_,background = 'tomato2',font = ("Copperplate Gothic Light", 10)).grid(column = 0,
              row = 4)
    
    self.skill_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/final_skill_data.csv')
    self.skill_data = self.skill_data[self.skill_data.Eid.isin(index)]
    
  def next(self):
    data = self.skill_data
    self.window.destroy()
    master = Tk()
    app = SixthPage(master,data)
    master.mainloop()      
    print("Sucess!!!")
    

class SixthPage:
  def __init__(self,master,data):
    self.data = data
    self.window = master
    self.window.title("Collaborative Filtering Result(Top 50 Employees)")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    self.l = ttk.Label(self.window, text = "Processing Wait",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)  
    self.b1 = ttk.Button(self.window,text="calculate",command=self.calculate).grid(row = 600, column = 5)
    self.b2 = ttk.Button(self.window,text="Next",command=self.next).grid(row = 600, column = 10)
  def calculate(self):
    rec1 = CollaborativeRecommender(self.data)
    rec1.create_pivot()
    rec1.create_datamatrix()
    rec1.Recommender()
    c_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/skill_details.csv')
    col = c_data.iloc[-1,:]
    list_ = []
    for i in col:
      if i == '0' or i == 0:
        list_.append(0)
      else:
        list_.append(5)
    print(list_)
    query = np.array(list_).reshape(1,-1)
    res = rec1.Recommend(query)
    res = res[0]
    data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/Final_Employees_Data.csv')
    self.newdata = data[data['Eid'].isin(res)]
    self.newdata = self.newdata[['Eid','Ename']]
    self.l = ttk.Label(self.window, text = "TOP 50 EMPLOYEES FROM COLLABORATIVE FILTERING",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)
    for i in range(len(self.newdata)):
        #text_ = 'Employee Id: '+str(res[i])+' Employee Name: '+str((self.newdata[self.newdata['Eid']==res[i]]['Ename']).values[0])
        text_= 'Employee Id: '+str(self.newdata.iloc[i]['Eid'])+' Employee Name: '+str(self.newdata.iloc[i]['Ename'])
        self.l1 = ttk.Label(self.window, text = text_,background = 'tomato2',anchor="e", justify=LEFT,font = ("Copperplate Gothic Light", 10)).grid(column = 0,row = i+1)
    sdata = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/final_data_for_contentrecom.csv')
    Eid = [i for i in range(1,1001)]
    sdata['Eid']= Eid
    self.sdata = sdata[sdata['Eid'].isin(res)]
  def next(self):     
    data = self.sdata
    self.window.destroy()
    master = Tk()
    app = SeventhPage(master,data)
    master.mainloop()      
    print("Sucess!!!")

class SeventhPage:
  def __init__(self,master,data):
    self.data = data
    self.window = master
    self.window.title("Content Based Recommendation Result (Top 20 Results)")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    self.l = ttk.Label(self.window, text = "Processing Wait",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0) 
    self.b1 = ttk.Button(self.window,text="calculate",command=self.calculate).grid(row = 600, column = 5)
    self.b2 = ttk.Button(self.window,text="Next",command=self.next).grid(row = 600, column = 10)    

  def calculate(self):
    self.rec2 = ContentBasedRecommender(self.data)
    c_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/final_details.csv')
    cols = []
    col = c_data.iloc[-1,:]
    for i in col:
        cols.append(i)
    print(cols)
    self.rec2.create_matrix(cols)
    self.rec2.create_sigmoid()
    eid = int(self.data.iloc[0]['Eid'])
    final= self.rec2.do_recommendation(eid)
    final= final
    data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/Final_Employees_Data.csv')
    self.newdata = data[data['Eid'].isin(final)]
    self.newdata = self.newdata[['Eid','Ename']]
    self.l = ttk.Label(self.window, text = "TOP 20 EMPLOYEES FROM CONTENT BASED RECOMMENDATION",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)    
    for i in range(len(self.newdata)):
        text_= 'Employee Id: '+str(self.newdata.iloc[i]['Eid'])+' Employee Name: '+str(self.newdata.iloc[i]['Ename'])
        #text_ = 'Employee Id: '+str(final[i])+' Employee Name: '+str((self.newdata[self.newdata['Eid']==final[i]]['Ename']).values[0])
        self.l1 = ttk.Label(self.window, text = text_,background = 'tomato2',anchor="e", justify=LEFT,font = ("Copperplate Gothic Light", 10)).grid(column = 0,row = i+1)
    d_data = pd.read_csv('C:/Users/Matrix/Desktop/Final_year_ project/Data Files/Employee_Designation.csv')
    self.d_data = d_data[d_data['Eid'].isin(final)]
    
  def next(self):
    data = self.d_data
    self.window.destroy()
    master = Tk()
    app = EightPage(master,data)
    master.mainloop()      
    print("Sucess!!!")      

class EightPage:
  def __init__(self,master,data):
    self.data = data
    self.window = master
    self.window.title("Final Result with Designation")
    self.window.geometry('800x800')
    self.window.configure(background = "tomato2")
    self.l = ttk.Label(self.window, text = "Processing Wait",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)
    print(len(self.data))
    self.b1 = ttk.Button(self.window,text="calculate",command=self.calculate).grid(row = 600, column = 5)
    self.b2 = ttk.Button(self.window,text="Close",command=self.close).grid(row = 600, column = 10)    
  def calculate(self):
    self.dm  = DesignationMatching(self.data)
    self.dm.assign_priority()
    self.str_ = self.dm.designation_matching()
    self.l = ttk.Label(self.window, text = "FINAL RECOMMENDATION WITH DESIGNATION",background = 'tomato2',font = ("Bernard MT Condensed", 20)).grid(column = 1,
              row = 0)    
    self.l = ttk.Label(self.window, text = self.str_,background = 'tomato2',anchor="e", justify=LEFT,font = ("Copperplate Gothic Light", 15)).grid(column = 0,
              row = 1)    
  def close(self):
      self.window.destroy()
      
   

def main():
  root = Tk()
  app = FirstPage(root)
  root.mainloop()

if __name__=='__main__':
  main()
