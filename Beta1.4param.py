import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import KFold,StratifiedKFold
import numpy as np
import pickle
from sklearn.metrics import log_loss,accuracy_score

#try xgboost
#original fucntion from SRK
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, \
     seed_val=0, early_stop = 20,num_rounds=10000, eta = 0.1,\
     max_depth = 6,cv_dict = None,verbose_eval=True,\
     subsample = 0.7,colsample_bytree =0.7):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = eta
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample_bytree
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y,feature_names=feature_names)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y,feature_names=feature_names)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist,\
        early_stopping_rounds=early_stop,evals_result = cv_dict,verbose_eval = verbose_eval)
    else:
        xgtest = xgb.DMatrix(test_X,feature_names=feature_names)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


#lodaing data
data_path = "../../kaggleData/2sigma/"
train_file = data_path + "processed_train_df.json"
test_file = data_path + "processed_test_df.json"
train_df = pd.read_json(train_file)
#test_df = pd.read_json(test_file)

xgb14featureFile = 'xgb14.feat'
fileObject = open(xgb14featureFile,'r')
features = pickle.load(fileObject)
fileObject.close()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

KF=StratifiedKFold(train_y,5,shuffle=True,random_state = 42)

md_dict = {}
for md in [3,4,5,6,7]:

  cv_scores = []
  cv_result = []
  result_dict ={}
  models = []

  for dev_index, val_index in KF: 
      dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:] 

      dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
      dev_y, val_y = train_y[dev_index], train_y[val_index]
      
      preds,model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features,\
             early_stop = 64,num_rounds=10000,eta = 0.1,max_depth=md,cv_dict = result_dict,verbose_eval=100)
             
      loss = log_loss(val_y, preds)
      cv_scores.append(loss)
      cv_result.append(result_dict)
      models.append(model)
      
      print "finding the best max_depth"
      print md
      print cv_scores
      print np.mean(cv_scores)
      
      md_dict[md]=np.mean(cv_scores)
      
mini = 10
mini_md = 0
for md in [3,4,5,6,7]:
   if md_dict[md] < mini:
        mini = md_dict[md]
        mini_md = md

ss_list  = [0.3,0.5,0.7]
csb_list = [0.3,0.5,0.7]
another_dict = {}

for ss in ss_list:
     for csb in csb_list:
         cv_scores = []
         cv_result = []
         models = []
         result_dict ={}
 
         for dev_index, val_index in KF: 
             dev_set, val_set = train_df.iloc[dev_index,:] , train_df.iloc[val_index,:]
 
             dev_X, val_X = dev_set[features].as_matrix(), val_set[features].as_matrix()
             dev_y, val_y = train_y[dev_index], train_y[val_index]
             
             preds,model = runXGB(dev_X, dev_y, val_X, val_y,feature_names=features,\
                    early_stop = 64,num_rounds=10000,eta = 0.1,max_depth=mini_md,\
                    subsample = ss,colsample_bytree = csb,\
                    cv_dict = result_dict,verbose_eval=100)
                    
             loss = log_loss(val_y, preds)
             cv_scores.append(loss)
             cv_result.append(result_dict)
             models.append(model)
             
             print "finding the best subsample and colsample_bytree"
             print (ss,csb)
             print cv_scores
             print np.mean(cv_scores)
             
             another_dict[(ss,csb)]=np.mean(cv_scores)

mini = 10
best_ss = 0
best_csb =0
for ss in ss_list:
    for csb in csb_list:
        if another_dict[(ss,csb)] < mini:
            mini = another_dict[(ss,csb)]
            best_ss = ss
            best_csb = csb

print "The best for max depth is :"
print mini_md
print "The best set for subsample and colsample_bytree is :"
print (best_ss,best_csb)


