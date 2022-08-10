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
stream_sz=100
asks=100
op_file="task1_op_.txt"

hashes=10
bitarray=[0 for _ in range(69997)]
m=69997

f=open(op_file,"w") 
f.write("Time,FPR\n")
f.write(str(0)+","+str(0.0)+"\n")

bx=BlackBox()
curr_strm=[]

p=random.sample(list(sympy.primerange(len(bitarray), 2*len(bitarray))),hashes)
a=random.sample([k for k in range(2*len(bitarray))],hashes)
b=random.sample([k for k in range(2*len(bitarray))],hashes)

def myhashs(s):
    res = []
    for i in range(hashes):
        s_conv=int(binascii.hexlify(s.encode('utf8')),16)
        hash_val= ((a[i]*s_conv+b[i])%p[i])%m
        res.append(hash_val)
    return res

for i in range(asks):
    strm=bx.ask(ip_file,stream_sz)

    if i!=0:
        fp=0
        tn=0
        for strm_val in strm:
            tn_flag=False
            hash_vals = myhashs(strm_val)
            for val in hash_vals:
                if bitarray[val]==0:
                    tn_flag=True
                    tn+=1
                    break
            if not tn_flag:
                fp+=1
        
        fpr = fp/(fp+tn)
        print(i,fpr)
        f.write(str(i)+","+str(fpr)+"\n")
        
        

    for strm_val in strm:
        hash_vals = myhashs(strm_val)
        for val in hash_vals:
            bitarray[val]=1
    

 
f.close()
end=time.time()
print("Duration:",end-start)