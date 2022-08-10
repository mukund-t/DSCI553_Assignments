import sys
import time
from itertools import combinations
from pyspark import SparkContext
import json
from collections import defaultdict

start = time.time()
sc = SparkContext.getOrCreate()
sc.setLogLevel('Error')

f_thresh = 7
ip_file = "ub_sample_data.csv"
# ip_file = "sample.csv"
op_file = "task2.1_op_.csv"
op_file1 = "task2.2_op_.csv"

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

edgs = usr_comb.map(lambda x: sorted(x)).filter(lambda x: filter_edge(x[0],x[1]))
edgs_ls = edgs.collect()
vertx = edgs.flatMap(lambda x: [(x[0],),(x[1],)]).distinct().collect()
adj_ls = defaultdict(list)

for edg in edgs_ls:
    adj_ls[edg[0]].append(edg[1])
    adj_ls[edg[1]].append(edg[0])

def bfs(node, adj_ls, bw_edges):
    vis ={}
    lvl ={}
    inc ={}
    node_val ={}
    for vert in vertx:
        vis[vert[0]]=False
        lvl[vert[0]]=0
        inc[vert[0]]=0
        node_val[vert[0]]=1
  
    q = []
    trav = []
    for i in range(len(vertx)):
        trav.append([])
    term = []
    q.append(node)
    lv=1
    lvl[node]=0
    vis[node]=True
    inc[node]=1
    node_val[node]=0

    while len(q)!=0:
        t = q.pop(0)
        fl=False

        for j in adj_ls[t]:
            if not vis[j]:
                fl=True
                q.append(j)
                vis[j]=True
                lvl[j]=lvl[t]+1
                inc[j]+=inc[t]
                trav[lvl[t]].append([t,j])
            elif vis[j] and (lvl[j]-lvl[t])==1:
                fl=True
                inc[j]+=inc[t]
                trav[lvl[t]].append([t,j])

        if not fl:
            term.append(t)
    
    for j in range(len(trav)-1,-1,-1):
        for edg in trav[j]:
            if edg[1] in term:
                fr = (inc[edg[0]]/inc[edg[1]]) * node_val[edg[1]]
                bw_edges[tuple(sorted([edg[0],edg[1]]))] += fr
                node_val[edg[0]] += fr
        for edg in trav[j]:
            if edg[1] not in term:
                fr = (inc[edg[0]]/inc[edg[1]]) * node_val[edg[1]]
                bw_edges[tuple(sorted([edg[0],edg[1]]))] += fr
                node_val[edg[0]] += fr
    
    return bw_edges


# bfs('c')
bw_edges = edgs.map(lambda x: ((x[0],x[1]),0)).collectAsMap()
for vert in vertx:
    bw_edges = bfs(vert[0],adj_ls,bw_edges)

bw_list = []
for edg in bw_edges.keys():
    bw_list.append([edg[0],edg[1],round(bw_edges[edg]/2,5)])
bw_list.sort(key=lambda x: (-x[2],x[0],x[1]))

f=open(op_file,"w")
for i in bw_list:
    f.write("('"+i[0]+"', '"+i[1]+"'),"+str(i[2])+"\n")
f.close()

# task 2.2
def find_conncomp(temp_adj_ls):
    
    conn_comp = []
    vis ={}
    for vert in vertx:
        vis[vert[0]]=False

    def dfs(traver, vert, vis):
        vis[vert]=True
        traver.append(vert)

        for v in temp_adj_ls[vert]:
            if vis[v] == False:
                traver = dfs(traver, v, vis)
        return traver

    for vert in vertx:
        if vis[vert[0]]==False:
            conn_comp.append(dfs([],vert[0],vis))
    return conn_comp

def modularity(conn_comp):
    mod = 0
    m = len(edgs_ls)

    for comp in conn_comp:
        for n1 in comp:
            for n2 in comp:
                if n1 in adj_ls[n2]:
                    mod = mod + 1 - ((len(adj_ls[n1])*len(adj_ls[n2])) / (2*m))
                else:
                    mod = mod - ((len(adj_ls[n1])*len(adj_ls[n2])) / (2*m))
    
    return mod / (2*m)

all_vrtx = [vert[0] for vert in vertx]
maxmod = modularity([all_vrtx])
temp_adj_ls = adj_ls
temp_bw_list = bw_list
final_conncomp = []
bw_edges_ = edgs.map(lambda x: ((x[0],x[1]),0)).collectAsMap()

while (temp_bw_list[0][2] != 0):
    high_bw = temp_bw_list[0][2]
    n1 = temp_bw_list[0][0]
    n2 = temp_bw_list[0][1]
    temp_adj_ls[n2].remove(n1)
    temp_adj_ls[n1].remove(n2)

    conn_comps = find_conncomp(temp_adj_ls)
    modul = modularity(conn_comps)

    bw_edges = bw_edges_.copy()
    for vert in vertx:
        bw_edges = bfs(vert[0],temp_adj_ls, bw_edges)
    bw_list = []
    for edg in bw_edges.keys():
        bw_list.append([edg[0],edg[1],bw_edges[edg]/2])
    bw_list.sort(key=lambda x: (-x[2],x[0],x[1]))

    temp_bw_list = bw_list

    if modul > maxmod:
        maxmod = modul
        final_conncomp = conn_comps

final_conncomp = [sorted(x) for x in final_conncomp]
final_conncomp.sort(key=lambda z: (len(z), z[0]))  

f=open(op_file1,"w")
for set_comms in final_conncomp:
    temp=""
    for each_node in set_comms:
        temp+="'"+each_node+"', "
    temp=temp[:-2]+"\n"
    f.write(temp)
f.close()

end = time.time()
print("Duration", end-start)