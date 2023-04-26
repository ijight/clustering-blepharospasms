import os 
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
from ts2vg import NaturalVG
from scipy.spatial.distance import euclidean
from networkx import to_numpy_array
from community import community_louvain
import scipy.io
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from Functions import singlechannel_clustering as sc

singlechannel_timeseries = []
datadir = "Singlechannel/PreviousData_v2/"
for each in os.listdir(datadir):
    singlechannel_timeseries.append(list(scipy.io.loadmat(datadir+each)["X"][0]))

TIMESTEP = 150
OVERLAP = 10
numSubject = 5

segments = []
cnt = 0
segment_idx = np.zeros((numSubject, 2))
for each in range(0, numSubject):
    TS = singlechannel_timeseries[each]
    tmp_segment = [TS[i:i + TIMESTEP] for i in range(0, len(TS) - TIMESTEP + 1, TIMESTEP - OVERLAP)]
    print(np.array(tmp_segment).shape)
    if len(segments) == 0:
        segments = np.array(tmp_segment)
    else:
        segments = np.concatenate((segments, tmp_segment))
    segment_idx[each][1] = len(segments)

print(np.array(segments).shape)

# TS = singlechannel_timeseries[0]
TRUNCATED = 0   #len(TS) % TIMESTEP

weightedGraph = sc.generate_similiarity_matrix_segment(segments)
partition = sc.generate_partitions(weightedGraph)

colors = sc.contrasting_colors(len(set(partition.values())))
print(colors)
print(set(partition.values()))

for each in range(0, numSubject):
    TS = singlechannel_timeseries[each]
    each_segment_idx = np.arange(int(segment_idx[each][0]),int(segment_idx[each][1]))
    majority_partition = {}
    cnt = 0
    for i in range(0, len(TS)):
        if i < TIMESTEP - OVERLAP:
            majority_partition[i] = partition[each_segment_idx[cnt]]
        elif i > len(TS) - TIMESTEP:
            majority_partition[i] = partition[each_segment_idx[cnt]]
        else:
            if cnt < 2:
                majority_partition[i] = partition[each_segment_idx[cnt]]
            else:
                tmp = {}
                for j in range(cnt-2, cnt+1):
                    p = partition[each_segment_idx[j]]  # get partition of current segment
                    if p not in tmp:
                        tmp[p] = 1
                    else:
                        tmp[p] += 1

                majority = max(tmp, key=tmp.get)
                majority_partition[i] = majority

        if (i+1) % OVERLAP == 0:
            cnt = cnt + 1
            # print((i, cnt))

    print(set(majority_partition.values()))

    plt.subplot(1, numSubject, int(each)+1)

    for i in range(len(TS) - TRUNCATED):
        plt.plot(i, TS[i], marker='o', markersize=1, color=colors[majority_partition[i]])

    del TS, majority_partition


# majority_partition = sc.majority_partition(partition, TS, TIMESTEP, OVERLAP)
# colors = sc.contrasting_colors(len(set(majority_partition.values())))

# draw the weighted graph
# pos = nx.spring_layout(weightedGraph)
# # color the nodes according to their partition
# cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
# nx.draw_networkx_nodes(weightedGraph, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(weightedGraph, pos, alpha=0.5)
plt.show()



