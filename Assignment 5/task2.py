from blackbox import BlackBox
import time
import os
import sympy
import sys
from pyspark import SparkContext
import random
import binascii

start = time.time()
sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')
random.seed(553)

ip_file="users.txt"
stream_sz=300
asks=100
op_file="task2_op_.txt"

hashes=70
m=598

f=open(op_file,"w") 
f.write("Time,Ground Truth,Estimation\n")

bx=BlackBox()
curr_strm=[]

p=random.sample(list(sympy.primerange(m,m+1000)),hashes)
a=random.sample([k for k in range(m)],hashes)
b=random.sample([k for k in range(m)],hashes)

def myhashs(s):
    res = []
    for i in range(hashes):
        s_conv=int(binascii.hexlify(s.encode('utf8')),16)
        hash_val= ((a[i]*s_conv+b[i])%p[i])%m
        hash_val=bin(hash_val)
        trailing_zs=len(hash_val)-len(hash_val.rstrip('0'))
        # print(hash_val,trailing_zs)
        res.append(2**trailing_zs)
    return res

total_c=0
for i in range(asks):
    num_arr = [0 for _ in range(hashes)]
    strm=bx.ask(ip_file,stream_sz)
    strm=list(set(strm))
    #num_arr = [0 for _ in range(len(strm))]

    all_hash = []
    for strm_val in strm:
        hashed_vals = myhashs(strm_val)
        all_hash.append(hashed_vals)
        for j in range(hashes):
            if hashed_vals[j]>num_arr[j]:
               num_arr[j]=hashed_vals[j]
    
    # for k in range(hashes):
    #     for j in range(len(strm)):
    #         if all_hash[j][k]>num_arr[j]:
    #             num_arr[j] = all_hash[j][k]

    grp_avgs = []
    cnt = 0
    for k in range(0,hashes,10):
        grp_avgs.append(sum(num_arr[k:k+10])//10)
    for k in grp_avgs:
        if k<len(strm) and (len(strm)-k)<(len(strm)-cnt):
            cnt=k
    print(cnt, len(strm))
    total_c+=cnt
    f.write(str(i)+","+str(len(strm))+","+str(cnt)+"\n")
    
print(total_c/(asks*stream_sz))

 
f.close()
end=time.time()
print("Duration:",end-start)