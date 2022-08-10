import time
import os
import sys
from pyspark import SparkContext
import math


input_path = "yelp_train.csv"
#input_path = "sample.csv"
val_path = "yelp_val.csv"
#val_path = "sample_test.csv"
output_path = "task21op_test.csv"

#input_path = sys.argv[1]
#output_path = sys.argv[2]


sc = SparkContext.getOrCreate()

start=time.time()


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
res = testdata.map(lambda x: predict_rating(x[0],x[1])).collect()

f = open(output_path,"w")
f.write("user_id, business_id, prediction\n")

for r in res:
    f.write(r[0][0]+","+r[0][1]+","+str(r[1])+"\n")
f.close()      

end = time.time()
#print(res)
print('Duration:',end-start)