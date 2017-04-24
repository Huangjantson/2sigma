
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold,StratifiedKFold
#import matplotlib.pyplot as plt


# In[2]:

from mochi import *
import pickle


# In[3]:

import datetime


# In[13]:

#try xgboost
#original fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,      seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1,     max_depth = 6,cv_dict = None,verbose_eval=True):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.3
    param['seed'] = seed_val
    param['nthread'] = 2
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y,feature_names=feature_names)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y,feature_names=feature_names)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,        early_stopping_rounds=early_stop,evals_result = cv_dict,verbose_eval = verbose_eval)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# In[4]:

data_path = "/home/raku/kaggleData/2sigma/xgb145/"
train_file = data_path + "xgb1.45-train.json"
test_file = data_path + "xgb1.45-test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

pickl_file = data_path+'xgb145features.pickle'
fileObject = open(pickl_file,'r') 
features=pickle.load(fileObject)   
fileObject.close()
print len(features)



# In[11]:

#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=KFold(len(train_df),5,shuffle=True,random_state = 2333)


# In[ ]:
cv_result=[]
cv_scores=[]
models=[]
i=0
for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
       #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    preds,model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features,\
           early_stop = None,num_rounds=4500,eta = 0.02,max_depth=4,cv_dict = result_dict,verbose_eval=100)

    loss = log_loss(val_y, preds)
    
    cv_scores.append(loss)
    cv_result.append(result_dict)
    models.append(model)
    i+=1
    print 'loss for the turn '+str(i)+' is '+str(loss)


# In[ ]:

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)

cvResult = CVstatistics(cv_result,'mlogloss')

meanTestError = cvResult.result.filter(like='test').mean(axis=1)

print meanTestError[meanTestError==np.min(meanTestError)]

