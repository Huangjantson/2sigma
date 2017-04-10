
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold
import re
import string


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
    param['min_child_weight'] =1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.5
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y,feature_names=feature_names)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y,feature_names=feature_names)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=200)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# In[2]:

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
    #nth -> n
    #nst -> n
    #nrd -> n
    #nnd -> n
    pattern = re.compile('(\d+)((th)|(st)|(rd)|(nd))')
    street = pattern.sub('\g<1>',street)
    #deal with the w 14 street => west 14 street
    pattern = re.compile('(w)(\s+)(\d+)')    
    street = pattern.sub('west \g<3>',street)
    #deal with the e....
    pattern = re.compile('(e)(\s+)(\d+)')    
    street = pattern.sub('east \g<3>',street)
    

    return street
    
def getStreetNumber(address):
    #get building id in the front, return -1 if their isn't
    pattern = re.compile('^([\d-]*)([\s]+)')
    try:
        number = pattern.search(address).group(1)
        return int(number)
    except:
        return -1

#from "this is a lit"s python version by rakhlin
def singleValueConvert(df1,df2,column,mini_rank = 15):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = mini_rank
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = mini_rank
    return df1, df2


# In[4]:

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
    print 'New feature added:'+new_feature

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
train_df["price_per_bath"] =  train_df["price"]*1.0/train_df["bathrooms"]
train_df["price_per_bed"] = train_df["price"]*1.0/train_df["bedrooms"]
train_df["bath_per_bed"] = train_df["bathrooms"]*1.0/train_df["bedrooms"]
train_df["price_per_room"] = train_df["price"]*1.0/(train_df["bedrooms"]+train_df["bathrooms"])

test_df["price_per_bath"] =  test_df["price"]*1.0/test_df["bathrooms"]
test_df["price_per_bed"] = test_df["price"]*1.0/test_df["bedrooms"]
test_df["bath_per_bed"] = test_df["bathrooms"]*1.0/test_df["bedrooms"]
test_df["price_per_room"] = test_df["price"]*1.0/(test_df["bedrooms"]+test_df["bathrooms"])

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

#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
test_df["street_name"] = test_df["street_address"].apply(proecessStreet)

train_df["street_number"] = train_df["street_address"].apply(getStreetNumber)
test_df["street_number"] = test_df["street_address"].apply(getStreetNumber)

#features_to_use.append("street_number")


# In[11]:

#dealing feature with categorical features 
"""
display_address 8826    
building_id        7585   =》many zeros in this feature
manager_id   3481
street_address 15358 =》will be 3800 if no numbers in it 
"""
categorical = ["display_address", "manager_id", "street_address","street_name"]
for f in categorical:
        if train_df[f].dtype=='object':
            singleValueConvert(train_df,test_df,f,15)
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)


# In[13]:

#hcc encoding 
KF=KFold(len(train_df),5,shuffle=True,random_state = 2017)
train_df =train_df.join(pd.get_dummies(train_df[u'interest_level']).astype(int))
i = 0
for f , s in KF:
    print i
    
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'manager_id','high',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'manager_id','medium',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'building_id','high',update_df =train_df)
    hcc_scoring(train_df.iloc[f],train_df.iloc[s],'building_id','medium',update_df =train_df)
    i+=1
    
features_to_use.append('hcc_building_id_high')
features_to_use.append('hcc_building_id_medium')
features_to_use.append('hcc_manager_id_high')
features_to_use.append('hcc_manager_id_medium')


# In[ ]:

#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=KFold(len(train_df),5,shuffle=True,random_state = 42)


# In[ ]:

features_to_use.append('manager_skill')


# In[ ]:

#running and getting the cv from xgboost
cv_scores = []
#K-FOLD already defined.If not ,use
#KF=KFold(len(train_X),5,shuffle=True,random_state = 42)
for dev_index, val_index in KF:
        #split the orginal train set into dev_set and val_set
        dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
            
        #apply the function for createing some featues
        dev_set, val_set =manager_skill_eval(dev_set,val_set)
        
        #filter the features
        dev_X, val_X = dev_set[features_to_use].as_matrix(), val_set[features_to_use].as_matrix()
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        
        preds, model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features_to_use)
        cv_scores.append(log_loss(val_y, preds))
        break
print(cv_scores)
print np.mean(cv_scores)

