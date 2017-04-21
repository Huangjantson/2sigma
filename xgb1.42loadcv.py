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
    param['nthread'] = 4
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
    
#lodaing data
data_path = "/home/raku/kaggleData/2sigma/xgb142/"
store = "/home/raku/kaggleData/2sigma/xgb142/"
train_file = data_path + "xgb1.42-train.json"
test_file = data_path + "xgb1.42-test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

feature_file = data_path+'xgb142features.pickle'
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
    """
    #=============================================================        
    #feature engineerings for the categorical features
    #fill substitute the small size values by their mean
    for f in ['display_address','manager_id','building_id','street_name']:
        dev_set,val_set  = singleValueConvert(dev_set,val_set,f,1)
    
    #kmeans grouping
    getCluster(dev_set,val_set,30)
    getCluster(dev_set,val_set,10)
    

    #K-FOLD evaluation for the statistic features
    skf=KFold(len(dev_set['interest_level']),5,shuffle=True,random_state = 42)
    #dev set adding manager skill
    for train,test in skf:
            performance_eval(dev_set.iloc[train,:],dev_set.iloc[test,:],feature='manager_id',k=5,g=10,
                           update_df = dev_set,smoothing=False)
            temporalManagerPerf_f(dev_set.iloc[train,:],dev_set.iloc[test,:],update_df = dev_set)


    performance_eval(dev_set,val_set,feature='manager_id',k=5,g=10,smoothing=False)
    temporalManagerPerf_f(dev_set,val_set)
        
        
    #statitstic
    for f in main_st_nf:
        #print f
        categorical_statistics(dev_set,val_set,'manager_id',f)
        categorical_statistics(dev_set,val_set,'cluster_id_10',f)
        categorical_statistics(dev_set,val_set,'cluster_id_30',f)
        categorical_statistics(dev_set,val_set,'house_type',f)
        categorical_size(dev_set,val_set,'manager_id')
        categorical_size(dev_set,val_set,'house_type')
    
    #for f in price_related:
    #    rank_on_categorical(dev_set,val_set,'house_type_30',f,random =None)

    
    #manager main location
    manager_lon_lat(dev_set,val_set)
    
    for f in categorical:
    
        if dev_set[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(dev_set[f])+list(val_set[f]))
            dev_set[f] = lbl.transform(list(dev_set[f].values))
            val_set[f] = lbl.transform(list(val_set[f].values))
    
    #============================================================
    #dev_set.to_csv('having_view.csv',index=False,encoding  = 'utf-8')
    
    """
    #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    
    """
    runXGB(dev_X, train_y, val_X, test_y=None, feature_names=None, \
    seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1, max_depth = 6)
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

