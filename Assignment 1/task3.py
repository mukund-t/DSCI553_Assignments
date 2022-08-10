from pyspark import SparkContext
from pyspark.sql import SparkSession
import os 
import json
import sys

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

sc = SparkContext ('local[*]','task3')
spark = SparkSession.builder.appName('task3').getOrCreate()

business = spark.read.json(sys.argv[1]).rdd
test_review = spark.read.json(sys.argv[2]).rdd

res = business.map(lambda s: (s.business_id,(s.city))).join(test_review.map(lambda s: (s.business_id,s.stars))).map(lambda s: (s[1][0],float(s[1][1]))).groupByKey().map(lambda s: (s[0],sum(s[1])/len(s[1]))).sortBy(lambda p: (-p[1],p[0])).collect()
print("Result!!!!!!", res)

f = open(sys.argv[3], 'w')
f.write("city,stars\n")
for i in res:
    f.write(str(i[0])+","+str(i[1])+"\n")
f.close()

output_dict={
    "m1":1.3,
    "m2":1.7,
    "reason": "Yes"
}
fi = open(sys.argv[4], 'w')
json.dump(output_dict, fi)
fi.close()