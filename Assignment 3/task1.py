import math
import collections
import time
from functools import reduce
from operator import add
import copy
import os
import sympy
import sys
from itertools import combinations
from pyspark import SparkContext
import itertools
import math
import random


input_path = "yelp_train.csv"
output_path = "task1_op_.csv"

#input_path = sys.argv[1]
#output_path = sys.argv[2]


start=time.time()
sc = SparkContext.getOrCreate()

b_usr = sc.textFile(input_path).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id').map(lambda s: (s[1],s[0]))

usr_ids = list(set(b_usr.map(lambda a:a[1]).collect()))
usr_ids = sorted(usr_ids)

usr_id_dict = {}
for idx,usr_id in enumerate(usr_ids):
    usr_id_dict[usr_id]=idx
    
b_usr = b_usr.map(lambda s:(s[0],usr_id_dict[s[1]]))
b_usr_grp = b_usr.groupByKey().mapValues(set)
b_usr_grp = b_usr_grp.sortByKey()
b_usr_grp_dict = b_usr_grp.collectAsMap()   


usr_total=len(usr_ids)
hashes=150
p=random.sample(list(sympy.primerange(usr_total, 2*usr_total-1)),hashes)
#a=random.sample(range(0,usr_total-1),hashes)
#b=random.sample(range(0,usr_total-1),hashes)
a=random.sample([k for k in range(usr_total)],hashes)
b=random.sample([k for k in range(usr_total)],hashes)
                   
def hash_values(businesses):
    temp=[]
    n=int(usr_total)
    for j in range(hashes):
        temp.append(list(map(lambda z:((a[j]*z+b[j])%p[j])%n,businesses)))

    return_list=[min(j) for j in temp]

    return return_list
    
r=2
bnds=75

def bands(signatures):
    temp = []
    
    for j in range(bnds):
        t = signatures[1][j*r:(j+1)*r]
        temp.append(((j,tuple(t)),[signatures[0]]))
    return temp
                
    

singature_bands = b_usr_grp.mapValues(lambda businesses: hash_values(businesses)).flatMap(lambda signatures: bands(signatures)).reduceByKey(lambda a,b:a+b).filter(lambda l: len(l[1])>1).flatMap(lambda x: list(combinations(x[1],2))).distinct()


def jacard_sim(candidate_pair):
    b1 = b_usr_grp_dict[candidate_pair[0]]
    b2 = b_usr_grp_dict[candidate_pair[1]]
    
    js = len(set(b1).intersection(set(b2))) / (len(set(b1))+len(set(b2))-len(set(b1).intersection(set(b2))))
    
    return float(js)

final_list = singature_bands.map(lambda x: (x[0], x[1], jacard_sim(x))).filter(lambda s:s[2]>=0.5).sortBy(lambda o: (o[0],o[1],o[2])).collect()


f=open(output_path,"w")
f.write("business_id_1, business_id_2, similarity\n")

for it in final_list:
    f.write(str(it[0])+","+str(it[1])+","+str(it[2])+"\n")

f.close()


end = time.time()
print(len(final_list))
print("Duration:", end-start)

[[b1,b2,sim],]