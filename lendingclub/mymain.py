import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X = train.append(test)
ntrain = train.shape[0]

X['Y'] = np.where(X.loan_status=='Fully Paid',0,1)
X.drop('loan_status',axis=1,inplace= True)
# X.drop('id',axis=1,inplace=True)
X['term'] = X['term'].apply(lambda s: np.int8(s.split()[0]))
X.drop('grade', axis=1, inplace=True)
X.drop(labels='emp_title', axis=1, inplace=True)

X['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
X['emp_length'].replace('< 1 year', '0 years', inplace=True)

def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
    
X['emp_length'] = X['emp_length'].apply(emp_length_to_int)

X['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
X['log_annual_inc'] = X['annual_inc'].apply(lambda x: np.log10(x+1))
X.drop('annual_inc', axis=1, inplace=True)
X.drop('title', axis=1, inplace=True)
X.drop(labels='zip_code', axis=1, inplace=True)
X['earliest_cr_line'] = X['earliest_cr_line'].apply(lambda s: int(s[-4:]))
X['fico_score'] = 0.5*X['fico_range_low'] + 0.5*X['fico_range_high']
X.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
X['log_revol_bal'] = X['revol_bal'].apply(lambda x: np.log10(x+1))
X.drop('revol_bal', axis=1, inplace=True)
X = pd.get_dummies(X, 
                   columns=['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'initial_list_status', 'application_type'], 
                   drop_first=True)

#split data
train = X[:ntrain]
test = X[ntrain:]

train_y = train['Y']

train_x = train.drop(columns = ['Y','id'],axis=1)
test_x = test.drop(columns = ['Y','id'],axis=1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

#model 1
pipeline_sgdlogreg = Pipeline([
    ('imputer', Imputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])
param_grid_sgdlogreg = {
    'model__alpha': [10**1],
    'model__penalty': ['l1']
}
grid_sgd= GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_sgd.fit(train_x, train_y)
y_prob = grid_sgd.predict_proba(test_x)
# print(log_loss(test['Y'], y_prob))

part1 = pd.DataFrame({'id': test['id'],'prob': y_prob[:,1]})
part1.to_csv('mysubmission1.txt',index=False)

#model2
pipeline_xgb = Pipeline([
    ('imputer', Imputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', lgb.LGBMClassifier(silent=False,
                                random_state=1))])

param_xgb = {
    'model__max_depth': [50],
    'model__learning_rate': [0.1],
    'model__n_estimators': [200]
}

grid_xgb = GridSearchCV(estimator=pipeline_xgb, param_grid=param_xgb, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=3, verbose=1, return_train_score=False)

grid_xgb.fit(train_x, train_y)
y_prob = grid_xgb.predict_proba(test_x) 

# print(log_loss(test['Y'], y_prob))
part1 = pd.DataFrame({'id': test['id'],'prob': y_prob[:,1]})
part1.to_csv('mysubmission2.txt',index=False)

#model3
pipeline_rfc = Pipeline([
    ('imputer', Imputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
param_grid_rfc = {
    'model__n_estimators': [50] # The number of randomized trees to build
}

grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=3, verbose=1, return_train_score=False)
grid_rfc.fit(train_x,train_y)
y_prob = grid_rfc.predict_proba(test_x)

# print(log_loss(test['Y'], y_prob))
part1 = pd.DataFrame({'id': test['id'],'prob': y_prob[:,1]})
part1.to_csv('mysubmission3.txt',index=False)
