import numpy as np 
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
import time

start=time.time()
ip_file = "hw6_clustering.txt"
total_clusters = 10
op_file = "out_work.txt"

full_data = []
cluster_set = []
total_lines = 0
with open(ip_file) as fl_obj:
    for lines in fl_obj:
        temp = []
        temp.append(int(lines.split(",")[0]))
        temp.append(int(lines.split(",")[1]))
        temp.append(np.array(list(map(float,lines.split(",")[2:]))))
        cluster_set.append(temp[1])
        total_lines+=1
        full_data.append(temp)
fl_obj.close()

full_points_idx = set([f[0] for f in full_data])
full_points_cs = [f[1] for f in full_data]

full_data = np.array(full_data)
# np.random.seed(10)
np.random.shuffle(full_data)
split_data = np.array_split(full_data,5)

rs =[]
ds =[]
rest_points = []

x=[inst[2] for inst in split_data[0]]
kmeans = KMeans(n_clusters = 10*total_clusters).fit(x)
kmeans_labels = kmeans.labels_
outlier_labels = [i for i,j in Counter(kmeans_labels).items() if j==1]
for i in range(len(kmeans_labels)):
    if kmeans_labels[i] in outlier_labels:
        rs.append(split_data[0][i]) 
    else:
        rest_points.append(split_data[0][i])

x=np.array([inst[2] for inst in rest_points])
x_idx = np.array([inst[0] for inst in rest_points])
n_dim=len(x[0])
kmeans_ = KMeans(n_clusters = total_clusters).fit(x)
kmeans_labels_ = kmeans_.labels_

ds_idx = [[] for i in range(total_clusters)]
for i in range(len(x_idx)):
    ds_idx[kmeans_labels_[i]].append(x_idx[i])

x_labels = np.append(x,np.array(kmeans_labels_).reshape((len(x),1)),1)
for i in range(total_clusters):
    x_cluster = x_labels[x_labels[:,-1]==i]
    nsum = np.sum(x_cluster[:,:-1],axis=0)
    n2sum = np.sum(np.square(x_cluster[:,:-1]), axis=0)
    ds.append([len(x_cluster),nsum,n2sum])


x = np.array([inst[2] for inst in rs])
x_idx = np.array([inst[0] for inst in rs])
new_total_clusters=int(1+len(rs)/2)
# new_total_clusters = 5*total_clusters
kmeans6 = KMeans(n_clusters = new_total_clusters).fit(x)
kmeans6_labels = kmeans6.labels_
cs_cluster_centers_ = kmeans6.cluster_centers_

rs6 = []
cs6 = []
rs6_labels = [i for i,j in Counter(kmeans6_labels).items() if j==1]
for i in range(len(kmeans6_labels)):
    if kmeans6_labels[i] in rs6_labels:
        rs6.append(rs[i]) 
    else:
        cs6.append(rs[i])

cs_idx = [[] for i in range(new_total_clusters)]
for i in range(len(x_idx)):
    cs_idx[kmeans6_labels[i]].append(x_idx[i])

cs_idx = [c for c in cs_idx if len(c)>1]

cs = []
x_labels = np.append(x,np.array(kmeans6_labels).reshape((len(x),1)),1)
for i in range(new_total_clusters):
    if i not in rs6_labels:
        x_cluster = x_labels[x_labels[:,-1]==i]
        nsum = np.sum(x_cluster[:,:-1],axis=0)
        n2sum = np.sum(np.square(x_cluster[:,:-1]), axis=0)
        cs.append([len(x_cluster),nsum,n2sum])

f=open(op_file, "w")
f.write("The intermediate results:\n")

no_ds = 0
no_cs = 0
for ss in range(len(ds)):
    no_ds += ds[ss][0]
for ss in range(len(cs)):
    no_cs += cs[ss][0]
f.write("Round 1: "+str(no_ds)+","+str(len(cs))+","+str(no_cs)+","+str(len(rs6))+"\n")

#step7
for idx__ in range(1,5):
    new_data = split_data[idx__]
    rest_points = []
    
    cl_th = 2*np.sqrt(len(split_data[0][0][2]))

    tot=0
    for i in range(len(new_data)):

        std_dev_ds = []
        for i___ in range(len(ds)):
            t = (ds[i___][2]/ds[i___][0]) - ((ds[i___][1]/ds[i___][0])**2)
            std_dev_ds.append(t)

        feat = np.array(new_data[i][2])
        min_dist = np.inf
        assigned_cl = -1

        for j in range(len(ds)):
            d = np.sqrt(np.sum((feat - (ds[j][1]/ds[j][0]))**2 / std_dev_ds[j]))
            
            if d < cl_th and d < min_dist:
            # if d<min_dist and d< (2*np.sqrt(d)):
                min_dist = d
                assigned_cl = j

        if min_dist != np.inf:
            ds[assigned_cl][0]+=1
            ds[assigned_cl][1]+=feat
            ds[assigned_cl][2]+=(feat**2)
            ds_idx[assigned_cl].append(new_data[i][0])
        else:
            rest_points.append(new_data[i])
    
    for i in range(len(rest_points)):

        feat = np.array(rest_points[i][2])
        min_dist = np.inf
        assigned_cl = -1

        std_dev_cs = []

        for i__ in range(len(cs)):
            t = (cs[i__][2]/cs[i__][0]) - ((cs[i__][1]/cs[i__][0])**2)
            std_dev_cs.append(t)

        for j in range(len(cs)):

            d = np.sqrt(np.sum((feat - (cs[j][1]/cs[j][0]))**2 / std_dev_cs[j]))
            
            if d < cl_th and d < min_dist:
            # if d<min_dist and d< (2*np.sqrt(d)):
                min_dist = d
                assigned_cl = j

        if min_dist != np.inf:
            cs[assigned_cl][0]+=1
            cs[assigned_cl][1]+=feat
            cs[assigned_cl][2]+=(feat**2)
            cs_idx[assigned_cl].append(rest_points[i][0])
        else:
            rs6.append(rest_points[i])

    # #step11
    print("RS before", len(rs6))
    if (new_total_clusters<=len(rs6)):
        x = np.array([inst[2] for inst in rs6])
        x_idx = np.array([inst[0] for inst in rs6])
        kmeans11 = KMeans(n_clusters = new_total_clusters).fit(x)
        kmeans11_labels = kmeans11.labels_
        rs11_labels = [i for i,j in Counter(kmeans11_labels).items() if j==1]

        rs_temp = []
        cs_temp = []
        for i in range(len(kmeans11_labels)):
            if kmeans11_labels[i] in rs11_labels:
                rs_temp.append(rs6[i]) 
            else:
                cs_temp.append(rs6[i])

        cs_idx_temp = [[] for i in range(new_total_clusters)]
        for i in range(len(x_idx)):
            cs_idx_temp[kmeans11_labels[i]].append(x_idx[i])

        cs_idx_temp = [c for c in cs_idx_temp if len(c)>1]

        for i in range(len(cs_idx_temp)):
            cs_idx.append(cs_idx_temp[i])

        x_labels = np.append(x,np.array(kmeans11_labels).reshape((len(x),1)),1)
        for i in range(new_total_clusters):
            if i not in rs11_labels:
                x_cluster = x_labels[x_labels[:,-1]==i]
                nsum = np.sum(x_cluster[:,:-1],axis=0)
                n2sum = np.sum(np.square(x_cluster[:,:-1]), axis=0)
                cs.append([len(x_cluster),nsum,n2sum])
        
        rs6=rs_temp
        print("RS After",len(rs6))

        std_dev_cs = []

        for i__ in range(len(cs)):
            t = (cs[i__][2]/cs[i__][0]) - ((cs[i__][1]/cs[i__][0])**2)
            std_dev_cs.append(t)

        #merge cs clusters
        # print("Before merge", len(cs),cs,cs_idx)

        merge_cs = []
        for m_ in range(len(cs)):
            fl_mrg = 0
            for n_ in range(m_+1,len(cs)):
                feat = np.array((cs[m_][1]/cs[m_][0]))
                d = np.sqrt(np.sum((feat - (cs[n_][1]/cs[n_][0]))**2 / std_dev_cs[n_]))
                
                if d < cl_th:
                # if d< (2*np.sqrt(d)):
                    for p_ in range(len(merge_cs)):
                        if m_ in merge_cs[p_] or n_ in merge_cs[p_]:
                            fl_mrg=1
                            merge_cs[p_].add(m_)
                            merge_cs[p_].add(n_)
                    if fl_mrg==0:
                        fl_mrg=1
                        merge_cs.append(set([m_,n_]))

        print("CS merge",merge_cs)
        clusters_merge = None
        if len(merge_cs)==0:
            clusters_merge = set([])
        else:
            clusters_merge = set.union(*merge_cs)

        cs_temp = []
        cs_idx_temp = []
        for i_idx in range(len(cs)):
            if i_idx not in clusters_merge:
                cs_temp.append(cs[i_idx])
                cs_idx_temp.append(cs_idx[i_idx])
        
        for set_merge in merge_cs:
            st = list(set_merge)
            temp1 = [0,[0 for i in range(len(cs[0][1]))],[0 for i in range(len(cs[0][1]))]]
            temp2 = []
            for cs__ in st:
                temp1[0] += cs[cs__][0]
                temp1[1] += cs[cs__][1]
                temp1[2] += cs[cs__][2]
                temp2.extend(cs_idx[cs__])

            cs_temp.append(temp1)
            cs_idx_temp.append(temp2)
        
        cs = cs_temp
        cs_idx = cs_idx_temp
    
    if idx__<4:
        no_ds = 0
        no_cs = 0
        for ss in range(len(ds)):
            no_ds += ds[ss][0]
        for ss in range(len(cs)):
            no_cs += cs[ss][0]
        f.write("Round "+str(idx__+1)+": "+str(no_ds)+","+str(len(cs))+","+str(no_cs)+","+str(len(rs6))+"\n")
        
        # print("After merge", len(cs),cs,cs_idx)

#merge cs with ds
cs_merged_ds = []
for m_ in range(len(cs)):
    std_dev_ds = []
    for i___ in range(len(ds)):
        t = (ds[i___][2]/ds[i___][0]) - ((ds[i___][1]/ds[i___][0])**2)
        std_dev_ds.append(t)

    feat = np.array((cs[m_][1]/cs[m_][0]))
    min_dist = np.inf
    assigned_cl = -1
    
    for j in range(len(ds)):

        d = np.sqrt(np.sum((feat - (ds[j][1]/ds[j][0]))**2 / std_dev_ds[j]))
        
        if d < cl_th and d < min_dist:
        # if d<min_dist and d< (2*np.sqrt(d)):
            min_dist = d
            assigned_cl = j

    if min_dist != np.inf:
        cs_merged_ds.append(m_)
        ds[assigned_cl][0]+=cs[m_][0]
        ds[assigned_cl][1]+=cs[m_][1]
        ds[assigned_cl][2]+=cs[m_][2]
        ds_idx[assigned_cl].extend(cs_idx[m_])

print("CS merged with ds", cs_merged_ds)

cs_temp = []
cs_idx_temp = []
for ss in range(len(cs)):
    if ss not in cs_merged_ds:
        cs_temp.append(cs[ss])
        cs_idx_temp.append(cs_idx[ss])

cs = cs_temp
cs_idx = cs_idx_temp

no_ds = 0
no_cs = 0
for ss in range(len(ds)):
    no_ds += ds[ss][0]
for ss in range(len(cs)):
    no_cs += cs[ss][0]
f.write("Round "+str(5)+": "+str(no_ds)+","+str(len(cs))+","+str(no_cs)+","+str(len(rs6))+"\n")
f.write("\n")
f.write("The clustering results:\n")

# for i in range(len(ds_idx)):
#     print(len(ds_idx[i]))

pred_pts = []
ds_set = []
for i in range(len(ds_idx)):
    for ds_pt in ds_idx[i]:
        pred_pts.append([ds_pt,i])
        ds_set.append(ds_pt)
    
# print(len(ds_set))

outlier_set = list(full_points_idx - set(ds_set))
# print("Outlier Set",outlier_set)
for i in outlier_set:
    pred_pts.append([i,-1])

pred_pts.sort(key=lambda m:m[0])

for ss in pred_pts:
    f.write(str(ss[0])+","+str(ss[1])+"\n")
f.close()

end = time.time()
print("Duration:", end-start)
print(normalized_mutual_info_score(full_points_cs, [i[1] for i in pred_pts]))