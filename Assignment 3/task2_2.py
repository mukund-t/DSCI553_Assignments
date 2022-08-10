import time
import sys
from pyspark import SparkContext
import math
import xgboost as xgb
import json
import os

folder_path = "data"
val_path = "data/yelp_val.csv"
output_path = "task2_2op_.csv"

input_path = folder_path + "/yelp_train.csv"
userfile = folder_path + "/user.json"
bsnessfile = folder_path + "/business.json"

#input_path = sys.argv[1]
#output_path = sys.argv[2]

sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')

start=time.time()

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

for i in usr_b_valdata:
    if i[0] not in usersfeat.keys() or i[1] not in bsnessfeat.keys():
        print(i)

# modl = xgb.XGBRegressor(objective = 'reg:linear', n_estimators=100, max_depth=5, n_jobs=-1)
# modl.fit(x_train,y_train)

# predrat = modl.predict(x_test)

# f = open(output_path,"w")
# f.write("user_id, business_id, prediction\n")

# for j in range(len(usr_b_valdata)):
#     f.write(usr_b_valdata[j][0]+","+usr_b_valdata[j][1]+","+str(predrat[j])+"\n")
# f.close() 

end=time.time()
print('Duration:',end-start)