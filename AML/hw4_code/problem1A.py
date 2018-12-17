import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import numpy as np
%matplotlib inline

data=pd.read_csv('euro.txt',sep='\t')
df = data.iloc[:,1:]

for i in ['single','complete','average']:
    Z=linkage(df,i)
    #plt.figure(figsize=(80,50))
    fig,axes = plt.subplots(1,1,figsize=(30,15))
    plt.title('Dendrogram for '+i+' link',fontsize=50)
    plt.xlabel('country',fontsize=25)
    fig = dendrogram(Z)
    axes.set_xticklabels(list(data['Country'].values),fontsize=25)
    plt.show()