import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from numpy import linalg

%matplotlib inline

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
train1 = unpickle('data_batch_1')
train2 = unpickle('data_batch_2')
train3 = unpickle('data_batch_3')
train4 = unpickle('data_batch_4')
train5 = unpickle('data_batch_5')

test = unpickle('test_batch')

#data training
df1=pd.DataFrame(train1[b'data'])
df2=pd.DataFrame(train2[b'data'])
df3=pd.DataFrame(train3[b'data'])
df4=pd.DataFrame(train4[b'data'])
df5=pd.DataFrame(train5[b'data'])
df_test=pd.DataFrame(test[b'data'])
# data test
df1_label=pd.DataFrame(train1[b'labels'])
df2_label=pd.DataFrame(train2[b'labels'])
df3_label=pd.DataFrame(train3[b'labels'])
df4_label=pd.DataFrame(train4[b'labels'])
df5_label=pd.DataFrame(train5[b'labels'])
df_test_label=pd.DataFrame(test[b'labels'])

df=df1.append([df2,df3,df4,df5])
df_label=df1_label.append([df2_label,df3_label,df4_label,df5_label])

#
df['labels'] = df_label
df_test['labels'] = df_test_label

#
train=df
test=df_test

X_train = np.array(train.iloc[:,:-1])
y_train = np.array(train.iloc[:,-1])

X_test = np.array(test.iloc[:,:-1])
y_test = np.array(test.iloc[:,-1])

# train_cat = train.iloc[:,:3072]
# train_cat['labels'] = df_label
train_cat = train.append(test)

# compute mean image against each category

error=[]
mean_20=[]
for i in range(10):
    
    mean_img = train_cat[train_cat['labels']==i].mean(0)
   
    bat = train_cat.ix[train_cat['labels']==i,:3072]

    

    pca=PCA(20).fit(bat)
   
    r= pca.transform(bat)
    
    p = pca.inverse_transform(r)
    
    val = p.mean(0)
    
    mean_20.append(val)

    val = ((np.array(bat)-p)**2).sum()/6000

    error.append(val)


plt.figure()   
plt.bar(range(10),error,tick_label=range(10))
plt.title('error of using first 20 PCA for each category')

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
# Principle coordinate analysis
# compute distance

d=np.empty([10,3072])
for i in range(10):
    d[i] = mean_20[i]
    
D = pairwise_distances(d)

#multidimensional scalling
model = MDS(2,dissimilarity='precomputed')
out = model.fit_transform(D)
plt.scatter(out[:,0],out[:,1],c=range(10))
plt.title('2D map of the means for each category')
label = ['airplane0','automobile1','bird2','cat3','deer4','dog5','frog6','horse7','ship8','truck9']
for i in range(10):
    plt.text(out[i,0], out[i,1], label[i],fontsize=15)
plt.show()
pd.options.display.float_format = '{:,.2f}'.format
pd.DataFrame(D)

from numpy import linalg
lst=[]
for i in range(10):
    a = train_cat.ix[train_cat['labels']==i,:3072]

    pca =PCA(20).fit(a)

    for j in range(10):
        b = train_cat.ix[train_cat['labels']==j,:3072]

        r= pca.transform(b)
        p = pca.inverse_transform(r)

        val = ((np.array(b)-p)**2).sum()/6000
        
        lst.append(val)
        
err_m = np.reshape(lst,[10,10])
err_m = pd.DataFrame(err_m)
v = err_m

lst=[]
for i in range(10):
    for j in range(10):
        a = (v.iloc[i,j]+v.iloc[j,i])/2
        lst.append(a)
va = np.reshape(lst,[10,10])
va = pd.DataFrame(va)
pd.options.display.float_format = '{:,.2f}'.format
va

model = MDS(2,dissimilarity='precomputed')
out = model.fit_transform(va)
plt.scatter(out[:,0],out[:,1],c=range(10))
plt.title('2D map based on similarity of each category')
label = ['airplane0','automobile1','bird2','cat3','deer4','dog5','frog6','horse7','ship8','truck9']
for i in range(10):
    plt.text(out[i,0], out[i,1],label[i],fontsize=15)
plt.show()