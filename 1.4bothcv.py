
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
    
def showFscore(model,normalize = True):
    factors = model.get_fscore()
    factor_list = []
    total = sum(factors.values())
    for key in factors:
        if normalize:
            factors[key] = factors[key]*1.0/total
        else:
            factors[key] = factors[key]
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

#add ranking for this function
def performance_eval(train_df,test_df,feature,k,smoothing=True,g=1,f=1,update_df =None,random = None):
    #target_num_map = {'High':2, 'Medium':1, 'Low':0}
    temp=pd.concat([train_df[feature],pd.get_dummies(train_df.interest_level)], axis = 1)         .groupby(feature).mean()
     
    new_feature = feature+'_perf'
    new_rank = feature+'_rank'
    new_nrank = feature+'_nrank'
    
    temp.columns = ['tempHigh','tempLow', 'tempMed']
    
    temp[feature+'_origin'] = temp['tempHigh']*2 + temp['tempMed']
    mean_values = temp.loc[:, feature+'_origin'].mean()

    temp['count'] = train_df.groupby(feature).count().iloc[:,1]
    if smoothing:
        temp["lambda"] = g / (g + np.exp((k - temp["count"] )/f))
        temp[new_feature] = temp["lambda"]*temp[feature+'_origin']+(1-temp["lambda"])*mean_values
    else:
        temp[new_feature] = temp[feature+'_origin']
        
    temp[new_rank]=temp[new_feature].rank()
    temp[new_nrank]=temp[new_rank]/temp['count']
    
    # Add uniform noise. Not mentioned in original paper.adding to each manager
    if random:
        temp[new_feature] *= np.random.uniform(1 - random, 1 + random, len(temp))     

    value = test_df[[feature]].join(temp, on=feature, how="left")[[new_feature,new_rank,new_nrank]].fillna(mean_values)
    
    if update_df is None: update_df = test_df
    if new_feature not in update_df.columns: update_df[new_feature] = np.nan
    if new_rank not in update_df.columns: update_df[new_rank] = np.nan
    if new_nrank not in update_df.columns: update_df[new_nrank] = np.nan

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
    train_df['cluster_id_'+str(k)]=map(lambda x,y: cluster.predict(np.array([x,y]).reshape(1,-1))[0]                            if ~(np.isnan(x)|np.isnan(y)) else -1,                           train_df['latitude'],train_df['longitude'])
    test_df['cluster_id_'+str(k)]=map(lambda x,y: cluster.predict(np.array([x,y]).reshape(1,-1))[0]                            if ~(np.isnan(x)|np.isnan(y)) else -1,                           test_df['latitude'],test_df['longitude'])
    
#setting the outliers to be nan. to be test
def processMap(df):
    for i in ['latitude', 'longitude']:
        Q1 = df[i].quantile(0.005)
        Q3 = df[i].quantile(0.995)
        #IQR = Q3 - Q1
        upper = Q3
        lower = Q1
        df.ix[(df[i]>upper)|(df[i]<lower),i] = np.nan
        #df.ix[:,i] =  df[i].round(3) 
    return 


# In[4]:

def manager_lon_lat(train_df,test_df):
    
    #adding the features about distance and location
    temp=train_df[['manager_id',"latitude", "longitude"]].dropna()
    mean_value = temp.groupby('manager_id')[["latitude", "longitude"]].mean().round(4)
    mean_value.columns = ['mlat','mlon']
    std_value = train_df.groupby('manager_id')[["latitude", "longitude"]].std()
    mstd = std_value[["latitude", "longitude"]].mean()
    std_value['latitude']=std_value['latitude'].fillna(mstd['latitude'])
    std_value['longitude']=std_value['longitude'].fillna(mstd['longitude'])
    #manager mean distance
    std_value['m_m_distance'] = map(lambda x,y:np.sqrt(x**2+y**2).round(4),                                    std_value['latitude'],std_value['longitude'])
    
    #value = pd.concat([mean_value,std_value])

    updateMTest = test_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[['mlat','mlon']].fillna(-1)
    updateDTest = test_df[['manager_id']].join(std_value, on='manager_id', how="left")['m_m_distance'].fillna(-1)
    updateMTrain = train_df[['manager_id']].join(mean_value, on = 'manager_id', how="left")[['mlat','mlon']].fillna(-1)
    updateDTrain = train_df[['manager_id']].join(std_value, on='manager_id', how="left")['m_m_distance'].fillna(-1)
    
    for f in ['mlat','mlon','m_m_distance']:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns: 
            train_df[f] = np.nan
    
    test_df.update(updateDTest)
    test_df.update(updateMTest)
    
    train_df.update(updateDTrain)
    train_df.update(updateMTrain)


# In[5]:

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


# In[10]:

#the new one not using cv-manner for the statistics
def categorical_statistics(train_df,test_df,cf,nf,                           get_median=True,get_min = True,get_max = True,                           get_normalized_in_group = True,mini_size = 20):
    statistics ={}
    statistics['mean']='mean'
    statistics['std']='std'
    statistics['size']='size'

    if get_max:
        statistics['max']='max'
    if get_min:
        statistics['min']='min'
    if get_median:
        statistics['median']='median'
        
    values = train_df.groupby(cf)[nf].agg(statistics)
    values = values.add_prefix(cf+'_'+nf+'_')
    
    new_feature = list(values.columns)
    
    #consider using -1 for others
    updateTest = test_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
    updateTrain = train_df[[cf]].join(values, on = cf, how="left")[new_feature]#.fillna(-1)
        
    for f in new_feature:
        if f not in test_df.columns: 
            test_df[f] = np.nan
        if f not in train_df.columns:
            train_df[f] = np.nan
    #update the statistics excluding the normalized value
    test_df.update(updateTest)
    train_df.update(updateTrain)



# In[8]:

def rank_on_categorical(train_df,test_df,cf,nf,mini_size=20,random=None):
    base = train_df.groupby(cf)[nf].agg({'rank':'rank','size':'size'})
    base['nrank'] = base['rank']/base['size']
    
    if mini_size:
        base.ix[base['size']<mini_size,:] = -1
    
    updateTrain = train_df[[cf]].join(base, on = cf, how="left").fillna(-1)
    updateTest = test_df[[cf]].join(base,on=cf,how = 'left').fillna(-1)

    n_feature = cf+'_'+nf+'_nrank'
    r_feature = cf+'_'+nf+'_rank'
    
    train_df[n_feature] =  updateTrain['rank']
    train_df[r_feature] =  updateTrain['nrank']
    
    test_df[n_feature] =  updateTest['rank']
    test_df[r_feature] =  updateTest['nrank']


# In[9]:

#try performance instead of high&medium
def temporalManagerPerf(train_df,test_df,update_df =None):
    temp=pd.concat([train_df,pd.get_dummies(train_df.interest_level)], axis = 1)
    tempTrain = temp[['manager_id','dayofyear','high','low','medium']].set_index('manager_id')
    tempTest = test_df[['manager_id','dayofyear']]
    tempJoin = tempTest.join(tempTrain,on='manager_id',how='left', rsuffix='_toSum')
    
    #3 day performance
    performance_3 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<4]
    performance_3 = performance_3.groupby(performance_3.index).sum()[['high','low','medium']]
    performance_3['total'] = performance_3['high']+performance_3['low']+performance_3['medium']
    performance_3['m3perf'] = (2*performance_3['high']+performance_3['medium'])*1.0/performance_3['total']

    
    performance_7 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<8]
    performance_7 = performance_7.groupby(performance_7.index).sum()[['high','low','medium']]
    performance_7['total'] = performance_7['high']+performance_7['low']+performance_7['medium']
    performance_7['m7perf'] = (2*performance_7['high']+performance_7['medium'])*1.0/performance_7['total']
    
    performance_14 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<15]
    performance_14 = performance_14.groupby(performance_14.index).sum()[['high','low','medium']]
    performance_14['total'] = performance_14['high']+performance_14['low']+performance_14['medium']
    performance_14['m14perf'] = (2*performance_14['high']+performance_14['medium'])*1.0/performance_14['total']

    
    performance_30 = tempJoin[tempJoin['dayofyear'] - tempJoin['dayofyear_toSum']<31]
    performance_30 = performance_30.groupby(performance_30.index).sum()[['high','low','medium']]
    performance_30['total'] = performance_30['high']+performance_30['low']+performance_30['medium']
    performance_30['m30perf'] = (2*performance_30['high']+performance_30['medium'])*1.0/performance_30['total']

    update = pd.concat([performance_3[['m3perf']],performance_7[['m7perf']],                        performance_14[['m14perf']],performance_30[['m30perf']]],axis=1).fillna(-1)

    if update_df is None: update_df = test_df
    
    new_features = ['m3perf','m7perf','m14perf','m30perf']
    
    for f in new_features:
        if f not in update_df.columns: 
             update_df[f] = np.nan
    
    update_df.update(update)
    


# In[11]:

#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[12]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[13]:

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


# In[14]:

#adding the house type
train_df['house_type']=map(lambda x,y:(x,y),train_df['bedrooms'],train_df['bathrooms'])
train_df['house_type'] = train_df['house_type'].apply(str)


# In[15]:

#filling outliers with nan
processMap(train_df)


# In[16]:

"""
new categorical data generated from the old ones
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
#test_df["street_name"] = test_df["street_address"].apply(proecessStreet)

train_df['building0']=map(lambda x:1 if x== '0' else 0,train_df['building_id'])
test_df['building0']=map(lambda x:1 if x== '0' else 0,test_df['building_id'])


# In[17]:

#dealing with features

#preprocessing for features
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')                                                             for i in x])
#test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')\
#                                                          for i in x])
#create the accept list
accept_list = list(featureList(train_df,test_df,limit = 0.001))

#map the feature to dummy slots
featureMapping(train_df,test_df,accept_list)
features_to_use.extend(map(lambda x : 'with_'+x,accept_list))


# In[12]:

#shorten reprocessing time: save the preprocessed train_df and test_df with some basic features
#train_df.to_json('train1.3std.json')
#test_df.to_json('test1.3std.json')
#print features_to_use
#shorten reprocessing time: load the preprocessed train_df and test_df with some basic features
#train_df=pd.read_json('train1.3std.json')
#test_df=pd.read_json('test1.3std.json')
#features_to_use = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'created_year', 'listing_id', 'created_month', 'created_day', 'created_hour', 'price_per_bed', 'bath_per_bed', 'price_per_room', u'with_exclusive', u'with_furnished', u'with_lowrise', u'with_common_parking/garage', u'with_pets_on_approval', u'with_terrace', u'with_live_in_superintendent', u'with_newly_renovated', u'with_full_time_doorman', u'with_duplex', u'with_dryer_in_unit', u'with_multi_level', u'with_garden', u'with_hardwood_floors', u'with_on_site_garage', u'with_fireplace', u'with_eat_in_kitchen', u'with_wifi_access', u'with_garage', u'with_subway', u'with_dining_room', u'with_view', u'with_publicoutdoor', u'with_hardwood', u'with_fitness_center', u'with_high_speed_internet', u'with_laundry_in_building', u'with_parking', u'with_garden/patio', u'with_prewar', u'with_on_site_laundry', u'with_valet', u'with_green_building', u'with_short_term_allowed', u'with_new_construction', u'with_reduced_fee', u'with_roofdeck', u'with_stainless_steel_appliances', u'with_simplex', u'with_dishwasher', u'with_washer_in_unit', u'with_cats_allowed', u'with_exposed_brick', u'with_roof_deck', u'with_common_outdoor_space', u'with_outdoor_areas', u'with_common_roof_deck', u'with_no_pets', u'with_childrens_playroom', u'with_central_a/c', u'with_wheelchair_access', u'with_post_war', u'with_renovated', u'with_elevator', u'with_highrise', u'with_loft', u'with_gym', u'with_luxury_building', u'with_outdoor_space', u'with_pre_war', u'with_residents_lounge', u'with_laundry_room', u'with_marble_bath', u'with_laundry_in_unit', u'with_parking_space', u'with_private_outdoor_space', u'with_high_ceiling', u'with_concierge', u'with_walk_in_closet(s)', u'with_doorman', u'with_balcony', u'with_dogs_allowed', u'with_gym/fitness', u'with_storage', u'with_live_in_super', u'with_lounge', u'with_granite_kitchen', u'with_private_balcony', u'with_laundry', u'with_actual_apt._photos', u'with_residents_garden', u'with_pool', u'with_washer/dryer', u'with_light', u'with_swimming_pool', u'with_high_ceilings', u'with_patio', u'with_no_fee', u'with_bike_room']

# In[18]:

#prepare for validation
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 42)

train_df = train_df.fillna(-1)
#test_df = test_df.fillna(-1)

train_df.to_csv('1.4cv-train-basic.csv',encoding='utf-8',index=False)
# In[33]:

print 'The fifth : house type and manager perf '
print '=================================================================='

#the basic features from preprocessing 
features = list(features_to_use)

#features to be added during cv by cv-manner statistics
features.extend(['manager_id_perf'])
features.extend(['m3perf','m7perf','m14perf','m30perf'])
features.extend(['manager_id_nrank'])


#categorical features to be added
categorical = ["display_address", "street_address","street_name",'building_id',\
'manager_id','building0','house_type']
features.extend(categorical)
features.extend(['cluster_id_10','cluster_id_30'])

#statistical features
features.extend(['m_m_distance','mlon','mlat'])

main_st_nf = ["bathrooms", "bedrooms","price_per_bed","bath_per_bed",\
"price_per_room","num_photos", "num_features", "num_description_words",'price']
main_statistics =['mean','max','min','median']

for st in main_statistics:
    features.extend(map(lambda x : 'manager_id_'+x+'_'+st,main_st_nf))
    features.extend(map(lambda x : 'house_type_'+x+'_'+st,main_st_nf)) 

features.extend(map(lambda x : 'cluster_id_10_'+x+'_'+'mean',main_st_nf))
features.extend(map(lambda x : 'cluster_id_30_'+x+'_'+'mean',main_st_nf))

price_related = ['price_per_bed','price_per_room','price']
#features.extend(map(lambda x : 'house_type_30_'+x+'_nrank',price_related))

features.extend(['manager_id_size','house_type_size'])


# In[20]:

features=list(set(features))


# In[37]:

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
            temporalManagerPerf(dev_set.iloc[train,:],dev_set.iloc[test,:],                           update_df = dev_set)
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
    
    for f in price_related:
        rank_on_categorical(dev_set,val_set,'house_type_30',f,random =None)

    
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
    print 'loss for the turn '+str(i)+' is '+str(loss)
    

# In[69]:
cvResult = CVstatistics(cv_result,'mlogloss')
meanTestError = cvResult.result.filter(like='test').mean(axis=1)


# In[70]:

print 'mean min test error:'+str(meanTestError[meanTestError==np.min(meanTestError)])

print 'average cv score:'
print np.mean(cv_scores)

