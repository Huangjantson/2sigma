
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing, ensemble
from sklearn.metrics import log_loss,accuracy_score
from sklearn.cross_validation import KFold,StratifiedKFold
import re
import string
from collections import defaultdict, Counter
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:

from mochi import *


# In[3]:

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
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
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

#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[5]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[6]:

#some transfromed features
# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
#test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
#test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
#test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
#test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
#test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
#test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
#test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
#test_df["created_hour"] = test_df["created"].dt.hour

#some new numerical features related to the price
train_df["price_per_bath"] =  (train_df["price"]*1.0/train_df["bathrooms"]).replace(np.Inf,-1)
train_df["price_per_bed"] = (train_df["price"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["bath_per_bed"] = (train_df["bathrooms"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["price_per_room"] = (train_df["price"]*1.0/(train_df["bedrooms"]+train_df["bathrooms"])).replace(np.Inf,-1)

#test_df["price_per_bath"] =  (test_df["price"]*1.0/test_df["bathrooms"]).replace(np.Inf,-1)
#test_df["price_per_bed"] = (test_df["price"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
#test_df["bath_per_bed"] = (test_df["bathrooms"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
#test_df["price_per_room"] = (test_df["price"]*1.0/(test_df["bedrooms"]+test_df["bathrooms"])).replace(np.Inf,-1)


# adding all these new features to use list # "listing_id",
features_to_use.extend(["num_photos", "num_features", "num_description_words",                        "created_year","listing_id", "created_month", "created_day", "created_hour"])
#price new features
features_to_use.extend(["price_per_bed","bath_per_bed","price_per_room"])

#for latter use
train_df["dayofyear"] = train_df["created"].dt.dayofyear
#test_df["dayofyear"] = test_df["created"].dt.dayofyear


# In[7]:

#adding the house type
train_df['house_type']=map(lambda x,y:(x,y),train_df['bedrooms'],train_df['bathrooms'])
train_df['house_type'] = train_df['house_type'].apply(str)


# In[8]:

#filling outliers with nan
processMap(train_df)


# In[9]:

"""
new categorical data generated from the old ones
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
#test_df["street_name"] = test_df["street_address"].apply(proecessStreet)

train_df['building0']=map(lambda x:1 if x== '0' else 0,train_df['building_id'])
test_df['building0']=map(lambda x:1 if x== '0' else 0,test_df['building_id'])


# In[10]:

#dealing with features

#preprocessing for features
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                             for i in x])
test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                         for i in x])
#create the accept list
accept_list = list(featureList(train_df,test_df,limit = 0.001))

#map the feature to dummy slots
featureMapping(train_df,test_df,accept_list)
features_to_use.extend(map(lambda x : 'with_'+x,accept_list))


# In[11]:

#prepare for validation
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 42)

train_df = train_df.fillna(-1)
#test_df = test_df.fillna(-1)


# In[16]:

#the basic features from preprocessing 
features = list(set(features_to_use))

#features to be added during cv by cv-manner statistics
features.extend(['manager_id_perf'])
features.extend(['m3perf','m7perf','m14perf','m30perf'])
features.extend(['manager_id_nrank'])


#categorical features to be added
categorical = ["display_address", "street_address","street_name",'building_id','manager_id','building0','house_type']
features.extend(categorical)
features.extend(['cluster_id_10','cluster_id_30'])


#statistical features
features.extend(['m_m_distance','mlon','mlat'])

main_st_nf = ["bathrooms", "bedrooms","price_per_bed","bath_per_bed","price_per_room","num_photos", "num_features", "num_description_words",'price']
main_statistics =['mean','max','min','median']

for st in main_statistics:
    features.extend(map(lambda x : 'manager_id_'+x+'_'+st,main_st_nf))
    features.extend(map(lambda x : 'house_type_'+x+'_'+st,main_st_nf)) 

features.extend(map(lambda x : 'cluster_id_10_'+x+'_'+'mean',main_st_nf))
features.extend(map(lambda x : 'cluster_id_30_'+x+'_'+'mean',main_st_nf))

price_related = ['price_per_bed','price_per_room','price']
#features.extend(map(lambda x : 'house_type_30_'+x+'_nrank',price_related))

features.extend(['manager_id_size','house_type_size'])


# In[ ]:

#to test
#features.extend(['cluster_id_10_d','cluster_id_30_d'])
#features.extend(['m3perf_f','m7perf_f','m14perf_f','m30perf_f'])


# In[13]:

features=list(set(features))


# In[14]:

#without leakage, size for train instead of both 
def categorical_size(train_df,test_df,cf):
    values =train_df.groupby(cf)['interest_level'].agg({'size':'size'})
    values = values.add_prefix(cf+'_')
    new_feature = list(values.columns)
    updateTest = test_df[[cf]].join(values, on = cf, how="left")[new_feature].fillna(-1)
    updateTrain = train_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
    
    for f in new_feature:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns:
            train_df[f] = np.nan
    #update the statistics excluding the normalized value
    test_df.update(updateTest)
    train_df.update(updateTrain)


# In[17]:

#running and getting the cv from xgboost
cv_scores = []
cv_result = []
models = []

i=0
for dev_index, val_index in KF: 
    result_dict = {}
    
    dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
    
    #=============================================================        
    """feature engineerings for the categorical features"""
    #fill substitute the small size values by their mean
    for f in ['display_address','manager_id','building_id','street_name']:
        dev_set,val_set  = singleValueConvert(dev_set,val_set,f,1)
    
    #kmeans grouping
    getCluster(dev_set,val_set,30)
    getCluster(dev_set,val_set,10)
    
    
    dev_set['house_type_30']=map(lambda x,y:(x,y),dev_set['house_type'],dev_set['cluster_id_30'])
    val_set['house_type_30']=map(lambda x,y:(x,y),val_set['house_type'],val_set['cluster_id_30'])
        
    dev_set['house_type_30'] = dev_set['house_type_30'].apply(str)
    val_set['house_type_30'] = val_set['house_type_30'].apply(str)

    #K-FOLD evaluation for the statistic features
    skf=KFold(len(dev_set['interest_level']),5,shuffle=True,random_state = 42)
    #dev set adding manager skill
    for train,test in skf:
            performance_eval(dev_set.iloc[train,:],dev_set.iloc[test,:],feature='manager_id',k=5,g=10,
                           update_df = dev_set,smoothing=False)
            temporalManagerPerf(dev_set.iloc[train,:],dev_set.iloc[test,:],update_df = dev_set)
            """
            #cv-manner statitstic
            for f in main_st_nf:
                #print f
                categorical_statistics(dev_set.iloc[train,:],dev_set.iloc[test,:],'manager_id',f,update_df=dev_set)
                categorical_statistics(dev_set.iloc[train,:],dev_set.iloc[test,:],'cluster_id_10',f,update_df=dev_set)
                categorical_statistics(dev_set.iloc[train,:],dev_set.iloc[test,:],'cluster_id_30',f,update_df=dev_set)
                #categorical_size(dev_set,val_set,'manager_id')
            """
            
            
    performance_eval(dev_set,val_set,feature='manager_id',k=5,g=10,smoothing=False)
    temporalManagerPerf(dev_set,val_set)
        
        
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
        
    #filter the features
    dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
    dev_y, val_y = train_y[dev_index], train_y[val_index]

    """
    runXGB(dev_X, train_y, val_X, test_y=None, feature_names=None, \
    seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1, max_depth = 6)
    """        
    
    preds,model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features,\
           early_stop = 64,num_rounds=10000,eta = 0.1,max_depth=4,cv_dict = result_dict,verbose_eval=100)

    loss = log_loss(val_y, preds)
    cv_scores.append(loss)
    cv_result.append(result_dict)
    models.append(model)
    i++
    print 'loss for the turn '+str(i)+' is '+str(loss)


# In[18]:

print 'The mean of the cv_scores (64 turns after best is:'
print np.mean(cv_scores)

