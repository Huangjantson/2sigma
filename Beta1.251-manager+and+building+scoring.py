
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


# In[2]:

#try xgboost
#fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None,      seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1, max_depth = 6):
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
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,        early_stopping_rounds=early_stop)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# In[3]:

#feature processing functions
#define punctutaion filter
def removePunctuation(x):
    #filter the head or tail blanks
    x = re.sub(r'^\s+',r' ',x)
    x = re.sub(r'\s+$',r' ',x)
    
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars, warning if you are dealing with other languages!!!!!!!!!!!!!!!
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    #change all the blank to space
    x = re.sub(r'\s',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    removing = string.punctuation#.replace('-','')# except '-'
    removed = re.sub("["+removing+"]", "", x)
    #removing the line-changing
    #removed = re.sub('\\n'," ",removed)    
    return removed

#feature processing functions
def proecessStreet(address):
    #remove the building number
    pattern = re.compile('^[\d-]*[\s]+')
    street = removePunctuation(pattern.sub('',address))
    
    #sub the st to street
    pattern = re.compile('( st)$')
    street = pattern.sub(' street',street)
    
    #sub the ave to avenue
    pattern = re.compile('( ave)$')
    street = pattern.sub(' avenue',street)
    
    pattern = re.compile('(\d+)((th)|(st)|(rd)|(nd))')
    street = pattern.sub('\g<1>',street)
    
    #deal with the w 14 street => west 14 street
    pattern = re.compile('(w)(\s+)(\d+)')    
    street = pattern.sub('west \g<3>',street)
    
    #deal with the e....
    pattern = re.compile('(e)(\s+)(\d+)')    
    street = pattern.sub('east \g<3>',street)
    
    return street
    
#from "this is a lit"s python version by rakhlin
def singleValueConvert(df1,df2,column,minimum_size=5):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= minimum_size, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= minimum_size, column] = -1
    return df1, df2



# In[4]:

def performance_eval(train_df,test_df,feature,k,g=1,f=1,update_df =None,random = None):
    target_num_map = {'High':2, 'Medium':1, 'Low':0}
    temp=pd.concat([train_df[feature],pd.get_dummies(train_df.interest_level)], axis = 1)         .groupby(feature).mean()
     
    new_feature = feature+'_perf'
    
    temp.columns = ['tempHigh','tempLow', 'tempMed']
    
    temp['count'] = train_df.groupby(feature).count().iloc[:,1]
    temp["lambda"] = g / (g + np.exp((k - temp["count"] )/f))
    temp[feature+'_origin'] = temp['tempHigh']*2 + temp['tempMed']
    mean_values = temp.loc[:, feature+'_origin'].mean()
    
    temp[new_feature] = temp["lambda"]*temp[feature+'_origin']+(1-temp["lambda"])*mean_values    
    
    # Add uniform noise. Not mentioned in original paper.adding to each manager
    if random:
        temp[new_feature] *= np.random.uniform(1 - random, 1 + random, len(temp))     

    value = test_df[[feature]].join(temp, on=feature, how="left")[new_feature].fillna(mean_values)
    
    if update_df is None: update_df = test_df
    if new_feature not in update_df.columns: update_df[new_feature] = np.nan
    update_df.update(value)


# In[5]:

"""the old manager skil """
#def manager_skill_eval(train_df,test_df,unrank_threshold = 10):
def manager_skill_eval(train_df,test_df,k,g=1,f=1,update_df =None,random = None):
    
    target_num_map = {'High':2, 'Medium':1, 'Low':0}
    temp=pd.concat([train_df['feature'],pd.get_dummies(train_df.interest_level)], axis = 1).groupby('manager_id').mean()
     
    temp.columns = ['tempHigh','tempLow', 'tempMed']
    
    temp['count'] = train_df.groupby('manager_id').count().iloc[:,1]
    temp["lambda"] = g / (g + np.exp((k - temp["count"] )/f))
    temp['manager_skill_origin'] = temp['tempHigh']*2 + temp['tempMed']
    mean_values = temp.loc[:, 'manager_skill_origin'].mean()
    
    temp['manager_skill'] = temp["lambda"]*temp['manager_skill_origin']+(1-temp["lambda"])*mean_values    
    
    # Add uniform noise. Not mentioned in original paper.adding to each manager
    if random:
        temp['manager_skill'] *= np.random.uniform(1 - random, 1 + random, len(temp))     

    value = test_df[['manager_id']].join(temp, on='manager_id', how="left")['manager_skill'].fillna(mean_values)
    
    if update_df is None: update_df = test_df
    if 'manager_skill' not in update_df.columns: update_df['manager_skill'] = np.nan
    update_df.update(value)
   


# In[6]:

#functions for features
def featureList(train_df,test_df,limit = 0.001):
    #acquiring the feature lists
    features_in_train = train_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    features_in_test = test_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    
    filtered_features_in_train = features_in_train[features_in_train > limit*len(train_df)]
    filtered_features_in_test = features_in_test[features_in_test > limit*len(test_df)]
    accept_list = set(filtered_features_in_train.index).union(set(filtered_features_in_test.index))
    return accept_list

def featureMapping(train_df,test_df,feature_list):
    for feature in feature_list:
        #add the feature column for both
        #if feature in the row, then set the value for (row,feature) to 1
        train_df['with_'+feature]=train_df['features'].apply(lambda x : 1 if feature in x else 0)
        test_df['with_'+feature]=test_df['features'].apply(lambda x : 1 if feature in x else 0)
    return


# In[7]:

#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[8]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[9]:

#some new numerical features related to the price
train_df["price_per_bath"] =  (train_df["price"]*1.0/train_df["bathrooms"]).replace(np.Inf,-1)
train_df["price_per_bed"] = (train_df["price"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["bath_per_bed"] = (train_df["bathrooms"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["price_per_room"] = (train_df["price"]*1.0/(train_df["bedrooms"]+train_df["bathrooms"])).replace(np.Inf,-1)

test_df["price_per_bath"] =  (test_df["price"]*1.0/test_df["bathrooms"]).replace(np.Inf,-1)
test_df["price_per_bed"] = (test_df["price"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
test_df["bath_per_bed"] = (test_df["bathrooms"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
test_df["price_per_room"] = (test_df["price"]*1.0/(test_df["bedrooms"]+test_df["bathrooms"])).replace(np.Inf,-1)

features_to_use.extend(["price_per_bed","bath_per_bed","price_per_room"])
#features_to_use.append('price_per_bed')


# In[10]:

#some transfromed features
# count of photos #
train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

# count of "features" #
train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

# count of words present in description column #
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# convert the created column to datetime object so as to extract more features 
train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])

# Let us extract some features like year, month, day, hour from date columns #
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

# adding all these new features to use list # "listing_id",
features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year","listing_id", "created_month", "created_day", "created_hour"])


# In[11]:

"""
new categorical data append and converting label dummies for future use
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
test_df["street_name"] = test_df["street_address"].apply(proecessStreet)


# In[12]:

#dealing with features

#preprocessing for features
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                             for i in x])
test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                          for i in x])
#create the accept list
accept_list = list(featureList(train_df,test_df,limit = 0.001))

#map the feature to dummy slots
featureMapping(train_df,test_df,accept_list)
features_to_use.extend(map(lambda x : 'with_'+x,accept_list))


# In[13]:

#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 42)

train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)


# In[14]:

features_to_use.extend(['manager_id_perf','building_id_perf'])
#using these categorical features for direct labeling
categorical = ["display_address", "street_address","street_name",]
features_to_use.extend(categorical)


# In[18]:

cv_scores = []

#mini_ranking = 15

for dev_index, val_index in KF:
        #split the orginal train set into dev_set and val_set
        dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
        
#====================================================================        
        """feature engineerings for the categorical features"""
        #fill substitute the small size values by their mean
        for f in ['display_address','manager_id','building_id','street_name']:
            dev_set,val_set  = singleValueConvert(dev_set,val_set,f,1)
        
        
        #K-FOLD evaluation for the manager skill
        
        skf=StratifiedKFold(dev_set['interest_level'],5,shuffle=True,random_state = 42)
        #dev set adding manager skill
        for feature in ['manager_id','building_id']:
            for train,test in skf:
                performance_eval(dev_set.iloc[train,:],dev_set.iloc[test,:],feature=feature,k=5,g=10,
                               update_df = dev_set,random = 0.01)
            
            performance_eval(dev_set,val_set,feature=feature,k=5,g=10,random = 0.01)
        

        for f in categorical:

            if dev_set[f].dtype=='object':
                #print(f)
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(dev_set[f])+list(val_set[f]))
                dev_set[f] = lbl.transform(list(dev_set[f].values))
                val_set[f] = lbl.transform(list(val_set[f].values))
        
#===================================================================
                
        #filter the features
        dev_X, val_X = dev_set[features_to_use].as_matrix(), val_set[features_to_use].as_matrix()
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        
        """
        runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, \
        seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1, max_depth = 6)
        """        
        
        preds, model = runXGB(dev_X, dev_y, val_X, val_y,\
        feature_names=features_to_use,early_stop=64,
        num_rounds = 20000, eta = 0.1,max_depth = 4)
    
        #using rf for feature choosing
        #model = ensemble.RandomForestClassifier(500,random_state = 42,class_weight='balanced')
        #model.fit(dev_X,dev_y)
        #pred_prob = model.predict_proba(val_X)
        #pred = model.predict(val_X)
            
        cv_scores.append(log_loss(val_y, preds))

print cv_scores
#print accuracy_score(val_y,pred)


