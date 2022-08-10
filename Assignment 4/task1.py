import sys
import time
from itertools import combinations
from pyspark import SparkContext
import json
import numpy as np
from pyspark.sql import SparkSession
from graphframes import GraphFrame


start = time.time()
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('task1').getOrCreate()

f_thresh = 7
ip_file = "ub_sample_data.csv"
op_file = "task1_op_.txt"

data = sc.textFile(ip_file).map(lambda l: l.split(",")).filter(lambda l: l[0]!='user_id')

usr_bs = data.map(lambda s: (s[0],s[1])).groupByKey().mapValues(set)
usr_bs_dict = usr_bs.collectAsMap()

usr_list = data.map(lambda s:s[0]).distinct().collect()
usr_comb = sc.parallelize(sorted(list(combinations(usr_list,2))))

def filter_edge(bs1,bs2):
    inters_bs = list(usr_bs_dict[bs1].intersection(usr_bs_dict[bs2]))

    if len(inters_bs)>=f_thresh:
        return True
    return False

edgs = usr_comb.filter(lambda x: filter_edge(x[0],x[1]))
edgs_ls = edgs.union(edgs.map(lambda x:(x[1],x[0]))).collect()
vertx = edgs.flatMap(lambda x: [(x[0],),(x[1],)]).distinct().collect()

vrtx = spark.createDataFrame(vertx,["id"])
edgs = spark.createDataFrame(edgs_ls,["src","dst"])

g = GraphFrame(vrtx,edgs)
comm = g.labelPropagation(maxIter=5)

comms = comm.rdd
comms = comms.map(lambda l: (l[1],l[0])).groupByKey().mapValues(list)
comms = comms.map(lambda l:sorted(l[1])).sortBy(lambda s: len(s)).collect()

max_l=0
for c in comms:
    if len(c)>max_l:
        max_l=len(c)
arr=[]
for i in range(0,max_l+1):
    arr.append([])
for c in comms:
    arr[len(c)].append(c)
for i in range(len(arr)):
    arr[i] = sorted(arr[i], key=lambda x: x[0])

f=open(op_file,"w")
for set_comms in arr:
    for comm in set_comms:
        temp=""
        for each_node in comm:
            temp+="'"+each_node+"', "
        temp=temp[:-2]+"\n"
        f.write(temp)
f.close()

end=time.time()
print("Duration",end-start)