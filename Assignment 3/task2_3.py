import time
import sys
from pyspark import SparkContext
import math
import xgboost as xgb
import json
import os


start=time.time()
sc = SparkContext.getOrCreate()

folder_path = "data"
val_path = "data/yelp_test_ans.csv"
output_path = "task2_3op_.csv"

input_path = folder_path + "/yelp_train.csv"
userfile = folder_path + "/user.json"
bsnessfile = folder_path + "/business.json"

#model based

data = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
usr_b_r = data.map(lambda s: ((s[0],s[1]),float(s[2])))

data_val = sc.textFile(val_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
usr_b_val = data_val.map(lambda s: (s[0],s[1]))
usr_b_valdata = usr_b_val.collect()

users = sc.textFile(userfile).map(lambda r: json.loads(r))
usersfeat = users.map(lambda x:(x['user_id'],(float(x['review_count']),float(x['average_stars'])))).collectAsMap()

bsness = sc.textFile(bsnessfile).map(lambda r: json.loads(r))
bsnessfeat = bsness.map(lambda x:(x['business_id'],(float(x['review_count']),float(x['stars'])))).collectAsMap()

def create_train(usr,bsness):
    rc_usr = usersfeat[usr][0]
    rc_bsness = bsnessfeat[bsness][0]
    avgstr_usr = usersfeat[usr][1]
    str_bsness = bsnessfeat[bsness][1]
    return [rc_usr,rc_bsness,avgstr_usr,str_bsness]

x_train = usr_b_r.map(lambda x: create_train(x[0][0],x[0][1])).collect()
y_train = usr_b_r.map(lambda x: x[1]).collect()

x_test = usr_b_val.map(lambda x: create_train(x[0],x[1])).collect()

modl = xgb.XGBRegressor(objective = 'reg:linear', n_estimators=100, max_depth=5, n_jobs=-1)
modl.fit(x_train,y_train)

predrat_modl = modl.predict(x_test)

#item based
data = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')
usr_b_r = data.map(lambda s: ((s[0],s[1]),float(s[2])))
b_r_avg = data.map(lambda s:(s[1],float(s[2]))).groupByKey().map(lambda x: (x[0],sum(x[1])/len(x[1])))

bs = data.map(lambda s: s[1]).distinct().collect()
usr_bs = data.map(lambda s: (s[0],s[1])).groupByKey().mapValues(list)
bs_usr = data.map(lambda s: (s[1],s[0])).groupByKey().mapValues(list)

bs_usr_dict = bs_usr.collectAsMap()
usr_bs_dict = usr_bs.collectAsMap()
usr_b_r_dict = usr_b_r.collectAsMap()
b_r_avg_dict = b_r_avg.collectAsMap()
  

    
def predict_rating(usr,bsness):
    
    if usr not in usr_bs_dict.keys():
        return [(usr,bsness),3.0]
    
    if bsness not in bs_usr_dict.keys():
        return [(usr,bsness),3.0]
        
    
    w = []
    rated_bs = usr_bs_dict[usr]
    for b in rated_bs:
        if b!=bsness:
            usr_c = list(set(bs_usr_dict[b]).intersection(set(bs_usr_dict[bsness])))
            
            nr = 0
            tmp_dr1 = 0
            tmp_dr2 = 0

            if len(usr_c)>1:
                for u in usr_c:
                    nr += (usr_b_r_dict[(u,b)]-b_r_avg_dict[b])*(usr_b_r_dict[(u,bsness)]-b_r_avg_dict[bsness])
                    tmp_dr1 += (usr_b_r_dict[(u,b)]-b_r_avg_dict[b])**2
                    tmp_dr2 += (usr_b_r_dict[(u,bsness)]-b_r_avg_dict[bsness])**2
                
                dr = math.sqrt(tmp_dr1)*math.sqrt(tmp_dr2)
                
                if nr==0 or dr==0:
                    w.append([0,0])
                elif nr<0 or dr<0:
                    t=1
                else:
                    pr_cr = nr/dr
                    w.append([pr_cr*usr_b_r_dict[(usr,b)],pr_cr])

            #code for handling no common users       
            else:
                diff = abs(b_r_avg_dict[b]-b_r_avg_dict[bsness])
                if 0<=diff<=1:
                    w.append([usr_b_r_dict[(usr,b)],1])
                elif 1<diff<=2:
                    w.append([0.5*usr_b_r_dict[(usr,b)],0.5])
                else:
                    w.append([0,0])



    if len(w)==0:
        return [(usr,bsness),3.0]

    w.sort(key=lambda x: -x[1])
    
    nr=0
    dr=0
    for vals in w[:100]:
        nr+= vals[0]
        dr+= abs(vals[1])
    
    pred_r = 0.0
    if nr==0 or dr==0:
        pred_r = 0.0
    else:
        pred_r = nr/dr
    
    return [(usr,bsness),pred_r]
            
testdata = sc.textFile(val_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id') 
res = testdata.map(lambda x: predict_rating(x[0],x[1]))
predrat_item = res.map(lambda x: x[1]).collect()

final_rat = []
for rat in zip(predrat_modl,predrat_item):
    final_rat.append((0.9*rat[0])+(0.1*rat[1]))

f = open(output_path,"w")
f.write("user_id, business_id, prediction\n")

for j in range(len(usr_b_valdata)):
    f.write(usr_b_valdata[j][0]+","+usr_b_valdata[j][1]+","+str(final_rat[j])+"\n")
f.close() 

end=time.time()
print('Duration:',end-start)