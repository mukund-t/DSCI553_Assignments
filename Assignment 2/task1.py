import math
import collections
import time
from functools import reduce
from operator import add
import copy
import os
import sys
from itertools import combinations
from pyspark import SparkContext

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

start       = time.time()
case        = 1 
support     = 9
input_path  = "./data/small2.csv"
output_path = "./output.txt"

# case = int(sys.argv[1])
# support = int(sys.argv[2])
# input_path = sys.argv[3]
# output_path = sys.argv[4]


sc = SparkContext ('local[*]','task1')

if case == 1:
    small1  = sc.textFile(input_path, 2).map(lambda l: l.split(",")).filter(lambda l: l[0] != 'user_id').map(lambda l: (l[0], l[1])).distinct()
    baskets = small1.groupByKey().map(lambda s: sorted(list(s[1])))

elif case == 2:
    small1  = sc.textFile(input_path, 2).map(lambda l: l.split(",")).filter(lambda l: l[0] != 'user_id').map(lambda l: (l[1], l[0])).distinct()
    baskets = small1.groupByKey().map(lambda s: sorted(list(s[1])))

total_baskets = baskets.count()
buckets       = 99

#genereate new candidate pairs
def gen_new_cand(freq_cand):

    if len(freq_cand)>0 and freq_cand is not None:

        n        = len(freq_cand)
        si       = len(freq_cand[0])
        new_cand = []

        for i in range(n-1):
            for j in range(i+1,n):
                if freq_cand[i][:-1] == freq_cand[j][:-1]:
                    possible_cand = tuple(sorted(list(set(freq_cand[j]).union(set(freq_cand[i])))))
                    temp = []
                    for subset_item in combinations(possible_cand, si):
                        temp.append(subset_item)
                    if set(temp).issubset(set(freq_cand)):
                        new_cand.append(possible_cand)
                else:
                    break 
    return new_cand

#find the frequent itemsets
def find_itemsets(subset_baskets, support, total_baskets):

    temp           = list(subset_baskets)
    ps             = math.ceil(support * len(list(temp)) / total_baskets)
    subset_baskets = temp
    
    subset_baskets = list(subset_baskets)
    candidates_all = collections.defaultdict(list)

    count_dict = collections.defaultdict(int)
    bit_map    = [0 for i in range(buckets+1)]
   
    for basket in subset_baskets:
        for item in basket:
            count_dict[item]+=1
        for each_pair in combinations(basket, 2):
            key = (int(list(each_pair)[0]) + int(list(each_pair)[1])) % buckets
            bit_map[key] += 1

    single_freq = sorted([k for k in count_dict.keys() if count_dict[k]>= ps])
    bit_map     = list(map(lambda value: True if value >= ps else False, bit_map))

    item_len = 1
    cand     = single_freq

    candidates_all[str(item_len)] = [tuple(item.split(",")) for item in single_freq]

    while len(cand) > 0 and cand is not None:
        item_len  += 1
        count_dict = collections.defaultdict(int)

        for basket in subset_baskets:
            basket = sorted(list(set(basket).intersection(set(single_freq))))

            if len(basket) >= item_len:
                if item_len == 2:
                    for each_pair in combinations(basket, item_len):
                        key = (int(list(each_pair)[0]) + int(list(each_pair)[1])) % buckets
                        if bit_map[key]:
                            count_dict[each_pair] += 1

                if item_len >= 3:
                    for items in cand:
                        if set(items).issubset(set(basket)):
                            count_dict[items]+=1

        freq_cand = sorted([k for k in count_dict.keys() if count_dict[k] >= ps])

        cand = gen_new_cand(freq_cand)
        if len(freq_cand) == 0:
            break

        candidates_all[str(item_len)] = list(freq_cand)

    yield reduce(lambda val1, val2: val1 + val2, candidates_all.values())
    

#count frequent itemsets
def count_frequent_itemset(subset_baskets, candidates):
    count_dict = collections.defaultdict(int)
    for pairs in candidates:
        if set(pairs).issubset(set(subset_baskets)):
            count_dict[pairs]+=1

    yield [tuple((k, v)) for k, v in count_dict.items()]


def change_format(items):

    result= ""
    idx = 1
    
    for item in items:
        if len(item) == 1:
            result += str("(" + str(item)[1:-2] + "),")

        elif len(item) != idx:
            result  = result[:-1]
            result  += "\n\n"
            idx     = len(item)
            result  += (str(item) + ",")
        else:
            result += (str(item) + ",")

    return result[:-1]


candidates = baskets.mapPartitions(lambda part: find_itemsets(subset_baskets = part, support = support, total_baskets = total_baskets)).flatMap(lambda pair: pair).distinct()
sorted_candidates = candidates.sortBy(lambda pair: (len(pair), pair)).collect()
frequent_itemsets = baskets.flatMap(lambda part: count_frequent_itemset(subset_baskets = part,candidates = sorted_candidates)).flatMap(lambda pair: pair).reduceByKey(lambda a,b: a + b).filter(lambda a: a[1] >= support).map(lambda a: a[0])
sorted_frequent_itemsets=frequent_itemsets.sortBy(lambda pair: (len(pair), pair)).collect()

with open(output_path, 'w') as op_file:
    str_result = 'Candidates:\n' + change_format(sorted_candidates) + '\n\n' \
                    + 'Frequent Itemsets:\n' + change_format(sorted_frequent_itemsets)
    op_file.write(str_result)
    op_file.close()

end = time.time()
print("Duration: "+ str(end - start))

