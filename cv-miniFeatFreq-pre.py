import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold
import re
import string
from collections import defaultdict, Counter

#try xgboost
#fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=10000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
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

#encoded by sorted value
def hcc_sorting(train_df,test_df,feature,label=None,randomize=None):
    """
    sort the hcc feature by their prior on label then encode
    the train df should be with its labels get dummied
    
    return the list of all the possible features in order
    """
    if label ==None:
        train_df['tempScore'] = 2*train_df['high']+train_df['medium']
    
    label = 'tempScore'
    #get dummies for the 
    grouped  = train_df.groupby(feature)[label].agg({'size':'size','mean':'mean'})
    #unrankedMean = grouped.ix[grouped['size']<unrank_threshold,'mean'].mean()
    #grouped.ix[grouped['size']<unrank_threshold,'mean'] = unrankedMean
    grouped = grouped.reset_index()

    #get the values for the test set
    test_groupBy=test_df[feature].value_counts().to_frame().reset_index()
    test_groupBy.columns = [feature,'testSize']

    #merge together and reset the mean
    totalFeature = grouped.merge(test_groupBy,on = feature, how='outer').fillna(0)

    rankedMean = np.mean(grouped['mean'])

    #reset those train size 0 to the ranked mean
    totalFeature.ix[totalFeature['size']==0,'mean']=rankedMean
    
    #add some random 
    if randomize : 
        totalFeature['mean'] *= np.random.uniform(1 - randomize, 1 + randomize, len(totalFeature))

    return list(totalFeature.sort('mean')[feature])

#functions for features
def featureList(train_df,test_df,limit = 0.01):
    #acquiring the feature lists
    features_in_train = train_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    features_in_train.sort(ascending  = False)
    features_in_test = test_df["features"].apply(pd.Series).unstack().reset_index(drop = True).dropna().value_counts()
    features_in_test.sort(ascending  = False)
    
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

def clean(s):
    x = s.replace("-", "")
    x = x.replace(" ", "")
    x = x.replace("twenty four hour", "24")
    x = x.replace("24/7", "24")
    x = x.replace("24hr", "24")
    x = x.replace("24-hour", "24")
    x = x.replace("24hour", "24")
    x = x.replace("24 hour", "24")
    
    x = x.replace("common", "cm")
    x = x.replace("concierge", "doorman")
    x = x.replace("bicycle", "bike")
    x = x.replace("private", "pv")
    x = x.replace("deco", "dc")
    x = x.replace("decorative", "dc")
    x = x.replace("onsite", "os")
    x = x.replace("outdoor", "od")
    x = x.replace("ss appliances", "stainless")
    
    x = x.replace("high_",'hi')
    x = x.replace("high",'hi')
    x = x.replace("hi_",'hi')
   
    x = x.replace("live","lv")
    x = x.replace("pre_","pre")
    x = x.replace("all","al")
    x = x.replace("flex","fl")
    x = x.replace("residents","rs")
    x = x.replace("close","cl")
    x = x.replace("duplex","dp")
    x = x.replace("share","sh")
    x = x.replace("newly","new")
    x = x.replace("on_site","os")
    
    x = x.replace("24_hour","24")
    x = x.replace("full_time","24")
    
    x = x.replace("")


    return x

def feature_hash(x):
    cleaned = clean(x, uniq)
    key = cleaned[:4].strip()
    return key
    
    #lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

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

"""
new categorical data append and converting label dummies for future use
"""
#new feature for the street_address, use them instead of the original one
train_df["street_name"] = train_df["street_address"].apply(proecessStreet)
test_df["street_name"] = test_df["street_address"].apply(proecessStreet)

train_df =train_df.join(pd.get_dummies(train_df[u'interest_level']).astype(int))


"""
preprocessing for the features
"""
train_df["features"] = train_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_') \
                                                            for i in x])
test_df["features"] = test_df["features"].apply(lambda x:["_".join(i.split(" ")).lower().strip().replace('-','_')\
                                                          for i in x])


#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=KFold(len(train_df),5,shuffle=True,random_state = 42)




#=====================================================
"""cross validation"""
#the functions to be test in the cross validation
features_to_use.append('manager_skill')
categorical = ["display_address", "manager_id", "building_id", "street_address","street_name"]
features_to_use.extend(categorical)

#running and getting the cv from xgboost

#K-FOLD already defined.If not ,use
#KF=KFold(len(train_X),5,shuffle=True,random_state = 42)

"""the validating parameter"""

result_file = 'cv-findFeatMin-simple-pre.log'
feat_min_list = [0.0001,0.0003,0.001,0.003,0.01,0.03,0.0,0.1]
mini_ranking = 15



for feat_min in feat_min_list:
    cv_scores = []
    for dev_index, val_index in KF:
            feature_list = list(features_to_use)
            
            #split the orginal train set into dev_set and val_set
            dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 
            
            #special feature engineering for the trainset
            
            
    #====================================================================        
            """feature engineerings for the categorical features"""
            
            dev_set, val_set =manager_skill_eval(dev_set,val_set,\
            unrank_threshold = mini_ranking)
            
            
            #fill substitute the small size values by their mean
            for f in categorical:
                dev_set,val_set  = singleValueConvert(dev_set,val_set,f,mini_ranking)
            
                if dev_set[f].dtype=='object':
                    #print(f)
                    lbl = preprocessing.LabelEncoder()
                    lbl.fit(hcc_sorting(dev_set,val_set,f))
                    dev_set[f] = lbl.transform(list(dev_set[f].values))
                    val_set[f] = lbl.transform(list(val_set[f].values))
                    
    #===================================================================
            """
            selecting 'features' features
            """

    
            preaccept_list = list(featureList(dev_set,val_set,limit = 0.0001))
            
            key2original = defaultdict(list)
            origin2key =  {}
            k = 4
            for f in preaccept_list:
                cleaned = clean(f)
                key = cleaned[:k].strip()
                key2original[key].append(f)
                origin2key[f] = key
            
            #combine some of the features
            dev_set["features"] = dev_set["features"].apply(lambda x : [origin2key[i] if i in origin2key else i for i in x ])
            val_set["features"] = val_set["features"].apply(lambda x : [origin2key[i] if i in origin2key else i for i in x ])
                    
            accept_list = list(featureList(dev_set,val_set,limit = feat_min))
            featureMapping(dev_set,val_set,accept_list)
            feature_list.extend(map(lambda x : 'with_'+x,accept_list))
            
            #filter the features
            dev_X, val_X = dev_set[feature_list].as_matrix(), val_set[feature_list].as_matrix()
            dev_y, val_y = train_y[dev_index], train_y[val_index]
            
            preds, model = runXGB(dev_X, dev_y, val_X, val_y)
            cv_scores.append(log_loss(val_y, preds))

    #putting the output to the file
    try:
        f = open(result_file,'a')
    
        f.write('the cv-score for minimum accept (combined) feature frequency with value '\
        +str(feat_min)+' is: '\
        +str(cv_scores)+'\n')
        
        f.write('while the mean of it is :'+str(np.mean(cv_scores))+'\n')
    finally:
        f.close()


