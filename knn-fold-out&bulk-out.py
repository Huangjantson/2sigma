# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:43:34 2017

@author: dell
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold,StratifiedKFold
import pickle
from sklearn.neighbors import KNeighborsClassifier as KN
    
#lodaing data
store = "/home/raku/kaggleData/2sigma/knn4/"
train_df=pd.read_json(store+'knn-train.json')
test_df=pd.read_json(store+'knn-test.json')


pickl_file = store+'knn-weigthed-features.pickle'
fileObject = open(pickl_file,'r') 
features= pickle.load(fileObject)   
fileObject.close()
print len(features)
# In[17]:

#running and getting the cv from xgboost
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 2333)

cv_scores=[]
i=0

for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
       #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    kn = KN(4)
    kn.fit(dev_X,dev_y)
    preds = kn.predict_proba(val_X)

    #save the pickles for futures use
    pickl_file = store+'knn4-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()

    loss = log_loss(val_y, preds)
    
    cv_scores.append(loss)
    i+=1
    print'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)


# In[17]:


#run for test


train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

et = KN(4)
et.fit(train_X,train_y)
preds = et.predict_proba(test_X)


out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_json(store+'kn4-bulk-out.json')

#out_df.to_csv("xgb_beta1point42-0.02.csv", index=False)
# In[17]:
store = "/home/raku/kaggleData/2sigma/knn8/"
cv_scores=[]
i=0

for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
       #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    kn = KN(8)
    kn.fit(dev_X,dev_y)
    preds = kn.predict_proba(val_X)

    #save the pickles for futures use
    pickl_file = store+'knn8-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()

    loss = log_loss(val_y, preds)
    
    cv_scores.append(loss)
    i+=1
    print'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)


# In[17]:


#run for test


train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

et = KN(8)
et.fit(train_X,train_y)
preds = et.predict_proba(test_X)


out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_json(store+'kn8-bulk-out.json')

#out_df.to_csv("xgb_beta1point42-0.02.csv", index=False)

# In[17]:
store = "/home/raku/kaggleData/2sigma/knn16/"
cv_scores=[]
i=0

for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
       #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    kn = KN(16)
    kn.fit(dev_X,dev_y)
    preds = kn.predict_proba(val_X)

    #save the pickles for futures use
    pickl_file = store+'knn16-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()

    loss = log_loss(val_y, preds)
    
    cv_scores.append(loss)
    i+=1
    print'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)


# In[17]:


#run for test


train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

et = KN(16)
et.fit(train_X,train_y)
preds = et.predict_proba(test_X)


out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_json(store+'kn16-bulk-out.json')

# In[17]:
store = "/home/raku/kaggleData/2sigma/knn32/"
cv_scores=[]
i=0

for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
       #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    kn = KN(32)
    kn.fit(dev_X,dev_y)
    preds = kn.predict_proba(val_X)

    #save the pickles for futures use
    pickl_file = store+'knn32-5fold-out-'+str(i)+'.pickle'
    fileObject = open(pickl_file,'wb') 
    pickle.dump(preds,fileObject)   
    fileObject.close()

    loss = log_loss(val_y, preds)
    
    cv_scores.append(loss)
    i+=1
    print'loss for the turn '+str(i)+' is '+str(loss)

print 'The mean of the cv_scores is:'
print np.mean(cv_scores)


# In[17]:


#run for test


train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

et = KN(32)
et.fit(train_X,train_y)
preds = et.predict_proba(test_X)


out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_json(store+'kn32-bulk-out.json')