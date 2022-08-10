# Method Description:
# I used XGBRegressor for this task. My goal was to incorporate as many features as possible from the dataset files availabe.
# Features from each dataset file:
# User - review_count, average_stars, sum of all compliments, sum of useful funny and cool, elite years and fans
# Business - review_count, stars, latitude, longitude, is_open and number of categories
# Photo - number of photos for each business
# Tip - number of likes by a bussiness, user pair
# Checkin - duration of time for each business
# For the model parameters, I ran a grid search to find the best parameters which will give the lowest RMSE for validation data.
# I found learning_rate, max_depth, reg_lambda (to prevent overfitting) and n_estimators as useful features.

# Error Distribution:
# >=0 and <1: 102156
# >=1 and <2: 32944
# >=2 and <3: 6151
# >=3 and <4: 792 
# >=4: 1

# RMSE: 0.9587791369744758

# Execution Time: 467.30023765563965





import time
import sys
from pyspark import SparkContext
import math
import xgboost as xgb
import json
import os
from operator import add
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error

folder_path = "data"
val_path = "data/yelp_val.csv"
output_path = "task2_2op_.csv"

input_path = folder_path + "/yelp_train.csv"
userfile = folder_path + "/user.json"
bsnessfile = folder_path + "/business.json"
photofile = folder_path + "/photo.json"
checkinfile = folder_path + "/checkin.json"
tipfile = folder_path + "/tip.json"

#input_path = sys.argv[1]
#output_path = sys.argv[2]

sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')

start=time.time()


data = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
usr_b_r = data.map(lambda s: ((s[0],s[1]),float(s[2])))

data_val = sc.textFile(val_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
usr_b_val = data_val.map(lambda s: (s[0],s[1],float(s[2])))
usr_b_valdata = usr_b_val.collect()

users = sc.textFile(userfile).map(lambda r: json.loads(r))
usersfeat = users.map(lambda x:(x['user_id'],(float(x['review_count']), float(x['average_stars']), int(x['useful'])+int(x['funny'])+int(x['cool']), int(x['compliment_hot'])+int(x['compliment_more'])+int(x['compliment_profile'])+int(x['compliment_cute'])+int(x['compliment_list'])+int(x['compliment_note'])+int(x['compliment_plain'])+int(x['compliment_cool'])+int(x['compliment_funny'])+int(x['compliment_writer'])+int(x['compliment_photos']), int(x['fans']), int(len(x["elite"]) if x["elite"] is not None else "0") ))).collectAsMap()

bsness = sc.textFile(bsnessfile).map(lambda r: json.loads(r))
bsnessfeat = bsness.map(lambda x:(x['business_id'],(float(x['review_count']), float(x['stars']), (float(x["longitude"])+180)/360 if x["longitude"] is not None else 0.5, (float(x["latitude"])+90)/180 if x["latitude"] is not None else 0.5, int(x['is_open']), int(len(x["categories"]) if x["categories"] is not None else "0") ))).collectAsMap()

photo = sc.textFile(photofile).map(lambda r: json.loads(r))
photofeat = photo.map(lambda x:(x['business_id'],1)).reduceByKey(add).collectAsMap()

tip = sc.textFile(tipfile).map(lambda r: json.loads(r))
tipfeat = tip.map(lambda x:((x['business_id'],x['user_id']), x["likes"])).reduceByKey(add).collectAsMap()

checkin = sc.textFile(checkinfile).map(lambda r: json.loads(r))
checkinfeat = checkin.map(lambda x:(x['business_id'], len(x["time"]) )).reduceByKey(add).collectAsMap()

def create_train(usr,bsness):
    feat_ar = []

    us_feat = usersfeat.get(usr,[])
    if len(us_feat) == 6:
        for i in range(6):
            feat_ar.append(usersfeat[usr][i])
    else:
        for _ in range(6):
            feat_ar.append(np.nan)

    bs_feat = bsnessfeat.get(bsness,[])
    if len(bs_feat) == 6:
        for i in range(6):
            feat_ar.append(bsnessfeat[bsness][i])
    else:
        for _ in range(6):
            feat_ar.append(np.nan)
    
    feat_ar.append(photofeat.get(bsness,np.nan))
    feat_ar.append(tipfeat.get((bsness,usr),np.nan))
    feat_ar.append(photofeat.get(bsness,np.nan))

    return feat_ar

x_train = usr_b_r.map(lambda x: create_train(x[0][0],x[0][1])).collect()
y_train = usr_b_r.map(lambda x: x[1]).collect()

x_test = usr_b_val.map(lambda x: create_train(x[0],x[1])).collect()
y_test = usr_b_val.map(lambda x: x[2]).collect()


modl =  xgb.XGBRegressor(objective = 'reg:linear', learning_rate=0.1, max_depth=5, n_estimators=700, reg_lambda=1.5, n_jobs=-1)

modl.fit(x_train,y_train)
predrat = modl.predict(x_test)

print("RMSE",mean_squared_error(y_test,predrat))

ab_diff=np.absolute(predrat-y_test)
e1,e2,e3,e4,e5=0,0,0,0,0

for i in range(len(ab_diff)):
    if ab_diff[i]<1 and ab_diff[i]>=0:
        e1+=1
    elif ab_diff[i]<2 and ab_diff[i]>=1:
        e2+=1
    elif ab_diff[i]<3 and ab_diff[i]>=2:
        e3+=1
    elif ab_diff[i]<4 and ab_diff[i]>=3:
        e4+=1
    elif ab_diff[i]<5 and ab_diff[i]>=4:
        e5+=1
print("Error Distribution")
print(e1)
print(e2)
print(e3)
print(e4)
print(e5)

f = open(output_path,"w")
f.write("user_id, business_id, prediction\n")

for j in range(len(usr_b_valdata)):
    f.write(usr_b_valdata[j][0]+","+usr_b_valdata[j][1]+","+str(predrat[j])+"\n")
f.close() 

end=time.time()
print('Duration:',end-start)