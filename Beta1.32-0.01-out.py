
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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:

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

class CVstatistics(object):
    
    """
    self.result : the result dataframe storing the cv results
    self.endpoint : the first ending point for the validations
    self.turns: the turns for each validation
    
    validCurve : plot the validation curve,stop at the first endpoint
    errorsAt: return the average errors at a certain turn
    """
    def __init__(self,result_dict,metric,k=5):
        self.metric = metric
        if type(result_dict) == pd.DataFrame:
            self.result = result_dict
        else:
            temp_dict = {}
            for phase in ['train','test']:
                for turn in range(k):
                    temp_dict[phase+str(turn)]=cv_result[turn][phase][metric]
                    self.result=pd.DataFrame(dict([ (key,pd.Series(v)) for key,v in temp_dict.iteritems()]))    
        
        self.endpoint =len(self.result.filter(like = 'train').dropna())
        
        self.turns = self.result.filter(like = 'test').            apply(lambda x : ~np.isnan(x)).cumsum(axis=0).iloc[len(self.result)-1,:]

    def validCurve(self,start = 0, stop_at_first = True):
        if stop_at_first:
            eout = self.result.iloc[start:,:].filter(like = 'test').dropna().mean(axis=1)
            ein =  self.result.iloc[start:,:].filter(like = 'train').dropna().mean(axis=1)
        else:
            eout = self.result.iloc[start:,:].filter(like = 'test').mean(axis=1)
            ein =  self.result.iloc[start:,:].filter(like = 'train').mean(axis=1)
        plt.plot(map(lambda x :x+start,range(len(eout))), eout,
        map(lambda x :x+start,range(len(ein))), ein)
        plt.xlabel("turn")
        plt.ylabel(self.metric)
        plt.title('Validation Curve')
        
        plt.show()
    
    def eoutCurve(self,stop_at_first = True):
        if stop_at_first:
            eout = self.result.iloc[start:,:].filter(like = 'test').dropna().mean(axis=1)
        else:
            eout = self.result.iloc[start:,:].filter(like = 'test').mean(axis=1)
        plt.plot(map(lambda x :x+start,range(len(eout))), eout)
        plt.xlabel("turn")
        plt.ylabel(self.metric)
        plt.title('Eout Curve')
        
        plt.show()
        
    def minAvgEout(self):
        meanTestError = cvResult.result.filter(like='test').mean(axis=1)
        return meanTestError[meanTestError==np.min(meanTestError)]
    
    def errorsAt(self,turn):
        eout = self.result.filter(like = 'test').loc[turn].mean()
        ein = self.result.filter(like = 'train').loc[turn].mean()
        return eout,ein
    
def xgbImportance(model,factor_name):
    factors = model.get_score(importance_type=factor_name)
    factor_list = []
    total = sum(factors.values())
    for key in factors:
        factors[key] = factors[key]*1.0/total
        factor_list.append((key,factors[key]))
    return sorted(factor_list,key=lambda x : x[1],reverse=True)
    


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

#new function for clustering
def getCluster(train_df,test_df,k):
    cluster = KMeans(k,random_state = 2333)
    cluster.fit(train_df[['latitude', 'longitude']].dropna())
    train_df['cluster_id_'+str(k)]=map(lambda x,y: cluster.predict(np.array([x,y]).reshape(1,-1))                            if ~(np.isnan(x)|np.isnan(y)) else -1,                           train_df['latitude'],train_df['longitude'])
    test_df['cluster_id_'+str(k)]=map(lambda x,y: cluster.predict(np.array([x,y]).reshape(1,-1))                            if ~(np.isnan(x)|np.isnan(y)) else -1,                           test_df['latitude'],test_df['longitude'])
    
#setting the outliers to be nan. to be test
def processMap(train_df,test_df):
    for i in ['latitude', 'longitude']:
        Q1 = train_df[i].quantile(0.005)
        Q3 = train_df[i].quantile(0.995)
        IQR = Q3 - Q1
        upper = Q3
        lower = Q1
        train_df.ix[(train_df[i]>upper)|(train_df[i]<lower),i] = np.nan
        test_df.ix[(test_df[i]>upper)|(test_df[i]<lower),i] = np.nan
        #df.ix[:,i] =  df[i].round(3) 
    return 


# In[4]:

#new feature : manager base distance and manager action scope
#new features to be extended
"""
features.extend(['m_mean_bathrooms','m_mean_bedrooms','m_mean_price','m_mean_price_per_bed',\
                 'm_mean_bath_per_bed','m_mean_price_per_room','m_mean_num_photos',\
                 'm_mean_num_features','m_mean_num_description_words'])
"""
def manager_statistics(train_df,test_df,update_df =None,random = None):
    
    #adding the features about distance and location
    temp=train_df[['manager_id',"latitude", "longitude"]].dropna()
    mean_value = temp.groupby('manager_id')[["latitude", "longitude"]].mean().round(4)
    mean_value.columns = ['mlat','mlon']
    std_value = train_df.groupby('manager_id')[["latitude", "longitude"]].std()
    mstd = std_value[["latitude", "longitude"]].mean()
    std_value['latitude']=std_value['latitude'].fillna(mstd['latitude'])
    std_value['longitude']=std_value['longitude'].fillna(mstd['longitude'])
    #manager mean distance
    std_value['m_m_distance'] = map(lambda x,y:np.sqrt(x**2+y**2).round(4),\
                                    std_value['latitude'],std_value['longitude'])
    
    if random:
        std_value['m_m_distance'] *= np.random.uniform(1 - random, 1 + random, len(std_value))
        mean_value['mlat'] *= np.random.uniform(1 - random, 1 + random, len(mean_value))
        mean_value['mlon'] *= np.random.uniform(1 - random, 1 + random, len(mean_value))

    updateM = test_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[['mlat','mlon']]
    updateD = test_df[['manager_id']].join(std_value, on='manager_id', how="left")['m_m_distance']
    
    if update_df is None: update_df = test_df
    for f in ['mlat','mlon','m_m_distance']:
        if f not in update_df.columns: 
            update_df[f] = np.nan
    
    update_df.update(updateD)
    update_df.update(updateM)
    
    #adding the features about other things
    other_feature = ['bathrooms','bedrooms','price',"price_per_bed","bath_per_bed",\
                     "price_per_room",'num_photos','num_features','num_description_words']
    
    mean_value = train_df.groupby('manager_id')[other_feature].mean()
    mean_value = mean_value.add_prefix('m_mean_')
    
    new_mean_feature = list(mean_value.columns)
    
    updateM = test_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[new_mean_feature]
    
    for f in new_mean_feature:
        if f not in update_df.columns: 
            update_df[f] = np.nan

    update_df.update(updateM)


# In[5]:

#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[6]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[7]:

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

#some new numerical features related to the price
train_df["price_per_bath"] =  (train_df["price"]*1.0/train_df["bathrooms"]).replace(np.Inf,-1)
train_df["price_per_bed"] = (train_df["price"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["bath_per_bed"] = (train_df["bathrooms"]*1.0/train_df["bedrooms"]).replace(np.Inf,-1)
train_df["price_per_room"] = (train_df["price"]*1.0/(train_df["bedrooms"]+train_df["bathrooms"])).replace(np.Inf,-1)

test_df["price_per_bath"] =  (test_df["price"]*1.0/test_df["bathrooms"]).replace(np.Inf,-1)
test_df["price_per_bed"] = (test_df["price"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
test_df["bath_per_bed"] = (test_df["bathrooms"]*1.0/test_df["bedrooms"]).replace(np.Inf,-1)
test_df["price_per_room"] = (test_df["price"]*1.0/(test_df["bedrooms"]+test_df["bathrooms"])).replace(np.Inf,-1)


# adding all these new features to use list # "listing_id",
features_to_use.extend(["num_photos", "num_features", "num_description_words",                        "created_year","listing_id", "created_month", "created_day", "created_hour"])
#price new features
features_to_use.extend(["price_per_bed","bath_per_bed","price_per_room"])

#filling outliers with nan
processMap(train_df,test_df)


# In[8]:

"""
new categorical data generated from the old ones
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
test_df["street_name"] = test_df["street_address"].apply(proecessStreet)


# In[9]:

#dealing with features

#preprocessing for features
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                             for i in x])
test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                         for i in x])
#create the accept list
accept_list = list(featureList(train_df,test_df,limit = 0.001))

#map the feature to dummy slots
featureMapping(train_df,test_df,accept_list)
features_to_use.extend(map(lambda x : 'with_'+x,accept_list))


# In[10]:

#prepare for validation
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 42)

train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)


# In[11]:

#the basic features from preprocessing 
features = list(features_to_use)

#features to be added during cv by cv-manner statistics
features.extend(['manager_id_perf'])
#categorical features to be added
categorical = ["display_address", "street_address","street_name",'building_id','manager_id']
features.extend(categorical)
features.extend(['cluster_id_10','cluster_id_30'])


# In[12]:

features.extend(['m_mean_bathrooms','m_mean_bedrooms','m_mean_price','m_mean_price_per_bed',                 'm_mean_bath_per_bed','m_mean_price_per_room','m_mean_num_photos',                 'm_mean_num_features','m_mean_num_description_words'])
features.extend(['m_m_distance','mlon','mlat'])


# In[58]:

features=list(set(features))


# In[19]:

#=============================================================        
"""feature engineerings for the categorical features"""
#fill substitute the small size values by their mean
for f in ['display_address','manager_id','building_id','street_name']:
    train_df,test_df  = singleValueConvert(train_df,test_df,f,1)


#K-FOLD evaluation for the statistic features

skf=StratifiedKFold(train_df['interest_level'],5,shuffle=True,random_state = 42)
#dev set adding manager skill
for feature in ['manager_id']:
    for train,test in skf:
        performance_eval(train_df.iloc[train,:],train_df.iloc[test,:],feature=feature,k=5,g=10,
                       update_df = train_df)
        manager_statistics(train_df.iloc[train,:],train_df.iloc[test,:],                          update_df = train_df)
    
    performance_eval(train_df,test_df,feature=feature,k=5,g=10)
    manager_statistics(train_df,test_df)

getCluster(train_df,test_df,30)
getCluster(train_df,test_df,10)

for f in categorical:

    if train_df[f].dtype=='object':
        #print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f])+list(test_df[f]))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))

#============================================================
        
#filter the features
train_X, test_X = train_df[features].as_matrix(), test_df[features].as_matrix()

preds,model = runXGB(train_X,train_y,test_X,feature_names=features,           num_rounds = 6400,eta = 0.01,max_depth=4,verbose_eval=100)

out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_beta1point32-0.01step.csv", index=False)


# In[18]:

train_X.shape

