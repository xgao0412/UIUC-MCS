import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans as mn
from sklearn.cluster import KMeans
import random
import glob
import random
from scipy.cluster.vq import vq
from scipy.cluster.vq import whiten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import math


def clf(seg_len=16,k1=40,k2 =12):
    random.seed(1)
    np.random.seed(1)
    
    path ='D:/MCS-DS/AML/hw4/notebook/HMP_Dataset/*/*.txt'
    files = glob.glob(path)
    
    bh=['brush_teeth','climb_stairs','comb_hair',
        'descend_stairs','drink_glass','eat_meat',
        'eat_soup','getup_bed','liedown_bed','pour_water',
        'sitdown_chair','standup_chair',
        'use_telephone','walk']
    
    behavior=bh
    
    tr=[]
    for i in bh:
        c=[k for k in files if i in k]
        ck = random.sample(c,math.floor(0.8*len(c)))
        for j in ck:
            tr.append(j)
    te=[]
    for i in files:
        if i not in tr:
            te.append(i)

    train_file=tr
    test_file=te



    lst=[]
    file_id=0


    for f in train_file:
        dt = pd.read_csv(f,sep=' ',names=['x','y','z'])

        for i in range(dt.shape[0]//seg_len):
            seg=np.array(dt[i*seg_len:(i+1)*seg_len]).flatten()
            seg_list = list(seg)

            seg_list.append(file_id)
    #       seg_list.append(f)
    #         seg = np.append(seg,np.array(file_id))
    #         seg= list(seg).append(f[-18:])
            seg_array=np.array(seg_list)
            lst.append(seg_array)

        file_id+=1

    train=pd.DataFrame(lst)

    # Hierarchical k means

    dr = 3*seg_len

    data=pd.DataFrame(train.iloc[:,:dr]).astype(None)
    labels = KMeans(k1,random_state=11).fit_predict(data)
    data['labels'] = labels

    lt=[]
    for i in range(k1):
        dt = data.ix[data['labels']==i,:dr]
        lt.append(pd.DataFrame(KMeans(k2,random_state=11).fit(dt).cluster_centers_))
    cluster = pd.concat(lt) 


    c=vq(data.iloc[:,:dr],cluster)[0]
    data['cluster']=c
    data['file'] = train[dr]

    #build train data
    feature=[]
    for i in range(len(train_file)):
        dt = data.ix[data['file']==i,'cluster']
        ca = np.histogram(dt,bins=np.arange((k1*k2+1)))[0]
        feature.append(ca)
    df_feature = pd.DataFrame(feature)

    df_feature['Y']=train_file

    for i in behavior:
        df_feature.ix[df_feature['Y'].str.contains(i),'Y']=i

    mapping = {key:value for value,key in enumerate(behavior)}
    df_feature['label'] = df_feature['Y'].map(mapping)
    df_feature['label'] = df_feature['label'].astype(int)

    #build test data
    lst=[]
    file_id=0

    for f in test_file:
        dt = pd.read_csv(f,sep=' ',names=['x','y','z'])

        for i in range(dt.shape[0]//seg_len):
            seg=np.array(dt[i*seg_len:(i+1)*seg_len]).flatten()
            seg_list = list(seg)

            seg_list.append(file_id)

            seg_array=np.array(seg_list)
            lst.append(seg_array)

        file_id+=1

    test=pd.DataFrame(lst)

    d=vq(test.iloc[:,:dr],cluster)[0]
    test['cluster']=d

    test_fe=[]
    for i in range(len(test_file)):
        dt = test.ix[test[dr]==i,'cluster']
        ca = np.histogram(dt,bins=np.arange((k1*k2+1)))[0]
        test_fe.append(ca)
    test_feature = pd.DataFrame(test_fe)

    df_test = pd.DataFrame(test_feature)
    df_test['Y']=test_file

    for i in behavior:
        df_test.ix[df_test['Y'].str.contains(i),'Y']=i

    mapping = {key:value for value,key in enumerate(behavior)}
    df_test['label'] = df_test['Y'].map(mapping)
    df_test['label'] = df_test['label'].astype(int)

    df_test = df_test.drop('Y',axis=1)
    df_train =df_feature.drop('Y',axis=1)

    X_train = df_train.iloc[:,:(k1*k2)]
    y_train = df_train['label']

    X_test = df_test.iloc[:,:(k1*k2)]
    y_test = df_test['label']



    clf = RandomForestClassifier(n_estimators=50,max_depth=30,random_state=0)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print((y_pred==y_test).sum()/len(y_test))

    return confusion_matrix(y_test,y_pred)




