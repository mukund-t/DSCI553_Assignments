from blackbox import BlackBox
import sys
import time
from pyspark import SparkContext
import random


start = time.time()
random.seed(553)
sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')

ip_file="users.txt"
stream_sz=100
asks=30
op_file="task3_op.txt"

bx=BlackBox()
curr_strm=[]
n=0
f=open(op_file,"w")
f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")

for i in range(asks):
    strm=bx.ask(ip_file,stream_sz)
    
    if i==0:
        curr_strm=strm
        n+=100
    else:
        for j in range(stream_sz):
            n+=1
            gen_prob=random.random()

            if (stream_sz/n) > gen_prob:
                rand_n = random.randint(0,99)
                curr_strm[rand_n]=strm[j]
    
    
    f.write(str(n)+","+curr_strm[0]+","+curr_strm[20]+","+curr_strm[40]+","+curr_strm[60]+","+curr_strm[80]+"\n")

f.close()

end=time.time()
print("Duration:",end-start)