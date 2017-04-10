
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import  preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold


# In[2]:

#try xgboost
#fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
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


# In[3]:

#lodaing data
data_path = "../"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
print(train_df.shape)
print(test_df.shape)


# In[4]:

#basic numerical features
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]


# In[5]:

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


# In[6]:

#dealing feature with categorical features 
"""
display_address 8826    
building_id        7585   =》many zeros in this feature
manager_id   3481
street_address 15358 =》will be 3800 if no numbers in it 
"""
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)


# In[7]:

#prepare for training
target_num_map = {'high':0, 'medium':1, 'low':2}

train_X = train_df[features_to_use]
test_X = test_df[features_to_use]

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=KFold(len(train_X),5,shuffle=True,random_state = 42)


# In[8]:

#running and getting the cv from xgboost
cv_scores = []
#K-FOLD already defined.If not ,use
#KF=KFold(len(train_X),5,shuffle=True,random_state = 42)
for dev_index, val_index in KF:
        dev_X, val_X = train_X.iloc[dev_index,:].as_matrix(), train_X.iloc[val_index,:].as_matrix()
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=list(list(train_X.columns)))
        cv_scores.append(log_loss(val_y, preds))
print(cv_scores)        


"""
# In[9]:

#ananlysis by the feature importance by weight
weight = model.get_score()
total = sum(weight.values())
for key in weight:
    weight[key] = weight[key]*1.0/total
weight


# In[10]:

#ananlysis by the feature importance by gain
gain = model.get_score(importance_type='gain')
total = sum(gain.values())
for key in gain:
    gain[key] = gain[key]*1.0/total
gain


# In[11]:

#ananlysis by the feature importance by coverage
cover = model.get_score(importance_type='cover')
total = sum(cover.values())
for key in cover:
    cover[key] = cover[key]*1.0/total
cover


# In[104]:

#output the outcome - using xgboost
train_X_m = train_X.as_matrix()
test_X_m = test_X.as_matrix()

preds, model = runXGB(train_X_m, train_y, test_X_m, num_rounds=400)
out_df = pd.DataFrame(preds)
out_df.columns = ["high", "medium", "low"]
out_df["listing_id"] = test_df.listing_id.values
out_df.to_csv("xgb_beta1.csv", index=False)

"""