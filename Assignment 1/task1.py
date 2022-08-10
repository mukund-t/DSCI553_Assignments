from pyspark import SparkContext
from pyspark.sql import SparkSession
import os 
import json
import sys

sc = SparkContext ('local[*]','task1')
spark = SparkSession.builder.appName('task1').getOrCreate()

# test_review = spark.read.json("../resource/asnlib/publicdata/test_review.json").rdd
test_review = spark.read.json(sys.argv[1]).rdd


output_dict = {}

output_dict["n_review"] = test_review.count()

output_dict["n_review_2018"] = test_review.map(lambda r: r.date.split("-")[0]).filter(lambda r: r == '2018').count()

output_dict["n_user"] = test_review.map(lambda r: r.user_id).distinct().count()

users = test_review.map(lambda r: (r.user_id,1)).reduceByKey(lambda a,b: a+b).sortBy(lambda r: (-r[1],r[0])).take(10)

businesses = test_review.map(lambda r: (r.business_id,1)).reduceByKey(lambda a,b: a+b).sortBy(lambda r: (-r[1],r[0])).take(10)

users_arr = []
businesses_arr = []
for i in users:
    temp = [i[0],i[1]]
    users_arr.append(temp)

for i in businesses:
    temp = [i[0],i[1]]
    businesses_arr.append(temp)

output_dict["top10_user"] = users_arr

output_dict["n_business"] = test_review.map(lambda r: r.business_id).distinct().count()

output_dict["top10_business"] = businesses_arr

# print(output_dict)

with open(sys.argv[2], 'w') as fp:
    json.dump(output_dict, fp)

fp.close()