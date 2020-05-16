# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:34:25 2019

@author: tbm1004
"""
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
from sklearn.cluster import KMeans

#in1 and in2 seem to be better for regression
#in3 and in4 is better for clustering

def Kmean(array):
    
    X = np.array(array)
    figs, axs = plt.subplots(3)
    axs[0].scatter(X[:,0],X[:,1], label='True Position')
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
    axs[1].scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
    axs[2].scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
    axs[2].scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
    
def run():
    g = input("Enter file name : ")
    file1 = open(g,"r+") 
    biglist = []
    for item in file1:
        mylist = []
        mylist.clear()
        item = item.replace('\n', '')
        x, y = item.split(",")
        mylist.append(float(x))
        mylist.append(float(y))
        biglist.append(mylist)
    Kmean(biglist)
    file1.close() 
    
if __name__ == '__main__':
    run()