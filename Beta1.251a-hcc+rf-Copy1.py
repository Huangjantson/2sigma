
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing, ensemble
from sklearn.metrics import log_loss,accuracy_score
from sklearn.cross_validation import KFold
import re
import string
from collections import defaultdict, Counter


# In[2]:

#try xgboost
#fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 4
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
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
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

def manager_skill_eval(train_df,test_df,unrank_threshold = 10):

    target_num_map = {'High':2, 'Medium':1, 'Low':0}
    temp=pd.concat([train_df.manager_id,pd.get_dummies(train_df.interest_level)], axis = 1).groupby('manager_id').mean()
     
    temp.columns = ['ManHigh','ManLow', 'ManMedium']
    
    print temp.columns
    temp['count'] = train_df.groupby('manager_id').count().iloc[:,1]
    
    temp['manager_skill'] = temp['ManHigh']*2 + temp['ManMedium']
    
    #ixes of the managers with to few sample
    unranked_managers_ixes = temp['count']<unrank_threshold
    ranked_managers_ixes = ~unranked_managers_ixes
    
    #test for using rank or unrank part for the filling values
    mean_values = temp.loc[unranked_managers_ixes, ['ManHigh','ManLow', 'ManMedium','manager_skill']].mean()
    mean_values_total = temp.loc[:, ['ManHigh','ManLow', 'ManMedium','manager_skill']].mean()
    
    #reset their values to their average
    temp.loc[unranked_managers_ixes,['ManHigh','ManLow', 'ManMedium','manager_skill']] = mean_values.values
    
    #assign the features for the train set
    new_train_df = train_df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
    
    #assign the features for the test/val set
    new_test_df = test_df.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
    new_manager_ixes = new_test_df['ManHigh'].isnull()
    new_test_df.loc[new_manager_ixes,['ManHigh','ManLow', 'ManMedium','manager_skill']] = mean_values_total.values           
    
    return new_train_df,new_test_df


# In[4]:

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


# In[5]:

def hcc_scoring(train_df,test_df,feature,labelValue,randomize=0.01,k=5,f=1,g=1,unrank_threshold =5,update_df =None):    
    #input is the train dataframe with its labels mapped to dummies
    #such as:
    #tempTrain = train_df.join(pd.get_dummies(train_df[u'interest_level']).astype(int))
    
    new_feature = '_'.join(['hcc',feature,labelValue])
    
    #take the mean  for the feature on the given featureValue which is mapped to dummies
    prob = train_df[labelValue].mean()
    
    #take the mean and count for each feature value
    grouped = train_df.groupby(feature)[labelValue].agg({'count':'size','mean':'mean'})
    
    #perform the transform for lambda and the final score
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["count"]) / f))
    grouped[new_feature] = grouped['lambda']*grouped['mean']+(1-grouped['lambda'])*prob
    
    #get the average score for the unrank features and reset them to this average
    unrankedMean = grouped.ix[grouped['count']<unrank_threshold,new_feature].mean()
    grouped.ix[grouped['count']<unrank_threshold,new_feature] = unrankedMean
    grouped = grouped.reset_index()
    
    #adding to the test_df
    update_value  = test_df[[feature]].merge(grouped,on = feature,how='left')[new_feature].fillna(prob)
    
    if randomize : update_value *= np.random.uniform(1 - randomize, 1 + randomize, len(test_df))
        
    #adding some noise to the new 
    #print 'New feature added:'+new_feature

    if update_df is None:
        update_df = test_df
    if new_feature not in update_df.columns: 
        update_df[new_feature] = np.nan
        
    update_df.update(update_value)
    return


# In[6]:

#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[7]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[8]:

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


# In[9]:

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


# In[10]:

"""
new categorical data append and converting label dummies for future use
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
test_df["street_name"] = test_df["street_address"].apply(proecessStreet)


# In[11]:

#dealing with features

#preprocessing for features
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                             for i in x])
test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                          for i in x])
#create the accept list
accept_list = list(featureList(train_df,test_df,limit = 0.001))

#map the feature to dummy slots
featureMapping(train_df,test_df,accept_list)
features_to_use.extend(map(lambda x : 'with_'+x,accept_list))


# In[12]:

#hcc encoding 
KF=KFold(len(train_df),5,shuffle=True,random_state = 2017)
train_df =train_df.join(pd.get_dummies(train_df[u'interest_level']).astype(int))
i = 0
for f , s in KF:
    #print i
    
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'manager_id','high',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'manager_id','medium',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'building_id','high',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'building_id','medium',update_df =train_df)
    i+=1
    
features_to_use.append('hcc_building_id_high')
features_to_use.append('hcc_building_id_medium')
features_to_use.append('hcc_manager_id_high')
features_to_use.append('hcc_manager_id_medium')


# In[13]:

train_df["created_weekday"] = train_df["created"].dt.dayofweek
test_df["created_weekday"] = test_df["created"].dt.dayofweek

train_df["is_weekend"] = train_df["created"].dt.dayofweek.apply(lambda x:x==5 or x==6)
test_df["is_weekend"] = test_df["created"].dt.dayofweek.apply(lambda x:x==5 or x==6)


#features_to_use.append('created_weekday')


# In[14]:

#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=KFold(len(train_df),3,shuffle=True,random_state = 42)

train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)


# In[15]:

features_to_use.append('manager_skill')
categorical = ["display_address", "manager_id", "building_id", "street_address","street_name"]
features_to_use.extend(categorical)


# In[22]:


cv_scores = []
in_sample = []

mini_ranking = 15

for dev_index, val_index in KF:
        #split the orginal train set into dev_set and val_set
        dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
        
        #special feature engineering for the trainset
        
        
#====================================================================        
        """feature engineerings for the categorical features"""
        
        dev_set, val_set =manager_skill_eval(dev_set,val_set,        unrank_threshold = mini_ranking)
        
        
        #fill substitute the small size values by their mean
        for f in categorical:
            dev_set,val_set  = singleValueConvert(dev_set,val_set,f,mini_ranking)
        
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
        
        #preds, model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features_to_use)
        
        #using rf for feature choosing
        model = ensemble.RandomForestClassifier(500,max_depth=35 \
        ,random_state = 42,class_weight='balanced',n_jobs = 2\
        ,min_samples_split=10)
        
        model.fit(dev_X,dev_y)
        pred_prob = model.predict_proba(val_X)
        orig_prob = model.predict_proba(dev_X)
        pred = model.predict(val_X)
            
        cv_scores.append(log_loss(val_y, pred_prob))
        in_sample.append(log_loss(dev_y, orig_prob))        
        #break
        
print 'in-sample error:'+str(in_sample)
print 'out-of sample error'+str(cv_scores)
print 'mean of in-sample'+str(np.mean(in_sample))
print 'mean of out-of sample error'+str(np.mean(cv_scores))
#print cv_scores
#print accuracy_score(val_y,pred)




