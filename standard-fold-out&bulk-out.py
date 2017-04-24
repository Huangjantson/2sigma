# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:43:34 2017

@author: dell
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold,StratifiedKFold
import pickle
from mochi import *
    
#lodaing data
data_path = "/home/raku/kaggleData/2sigma/xgb145/"
store = "/home/raku/kaggleData/2sigma/xgb142/"
train_file = data_path + "xgb1.42-train.json"
test_file = data_path + "xgb1.42-test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

feature_file = data_path+'xgb145features.pickle'
fileObject = open(feature_file,'r') 
features = pickle.load(fileObject)
fileObject.close()

# In[17]:

#running and getting the cv from xgboost
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 2333)

cv_scores = []
cv_result = []
models = []

i=0
for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 

    #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    
    
    """
    run model
    """        
    
    preds,model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features,\
           early_stop = None,num_rounds=3800,eta = 0.02,max_depth=4,cv_dict = result_dict,verbose_eval=100)

    loss = log_loss(val_y, preds)
    
    
    #save the pickles for futures use
    pickl_file = store+'xgb142-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()
    
    
    cv_scores.append(loss)
    cv_result.append(result_dict)
    models.append(model)
    i+=1
    print 'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)

cvResult = CVstatistics(cv_result,'mlogloss')

meanTestError = cvResult.result.filter(like='test').mean(axis=1)

print meanTestError[meanTestError==np.min(meanTestError)]

# In[17]:

"""
run for testing

train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

preds, model = runXGB(train_X, train_y, test_X,\
feature_names=features,
num_rounds = 3800, eta = 0.02,max_depth = 4,verbose_eval=100)

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df.to_json(store+'xgb142-bulk-out.json')
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_beta1point42-0.02.csv", index=False)
"""
