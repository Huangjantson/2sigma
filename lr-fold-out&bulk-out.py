# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:43:34 2017

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.linear_model import LogisticRegression
    
#lodaing data
data_path = "/home/raku/kaggleData/2sigma/lr4/"
store= "/home/raku/kaggleData/2sigma/lr4/"
train_file = data_path + "lr4-n-train.json"
test_file = data_path + "lr4-n-test.json"
print 'start loading'
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print train_df.shape
print test_df.shape

feature_file = data_path+'lr4features.pickle'
fileObject = open(feature_file,'r') 
features = pickle.load(fileObject)
fileObject.close()
print len(features)

# In[17]:

#running and getting the cv from xgboost
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(5,shuffle=True,random_state = 2333)



cv_scores = []
i=0        


for dev_index, val_index in KF.split(train_df,train_y): 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 

    #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    """
    run model
    """        
    
    lr = LogisticRegression(random_state=0)
    lr.fit(dev_X,dev_y)
    preds = lr.predict_proba(val_X)

    loss = log_loss(val_y, preds)
    
    
    #save the pickles for futures use
    pickl_file = store+'lr-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()
    
    cv_scores.append(loss)
    i+=1
    print 'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)

# In[17]:


train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

lr = LogisticRegression()
lr.fit(train_X,train_y)
preds = lr.predict_proba(test_X)

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_json(store+'lr-bulk-out.json')
