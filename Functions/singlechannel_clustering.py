import scipy
import numpy as np
import matplotlib.pyplot as plt
import os 
import networkx as nx
import igraph as ig
from ts2vg import NaturalVG
from scipy.spatial.distance import euclidean
from networkx import to_numpy_array
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# convert time series into natural visibility graphes
# https://github.com/CarlosBergillos/ts2vg

def construct_visibilty_graph(ts):
    # builds natural visibility graph
    vg = NaturalVG()
    vg.build(ts)
    # converts to networkx graph format
    g = vg.as_networkx()
    return g

def get_degree_sequence(vg):
    # get [(node, degrees),...] from a vg
    # convert to dict
    # get values (the degrees)
    ds = list(dict(nx.degree(vg)).values())
    return ds

def generate_similiarity_matrix(TS, TIMESTEP, OVERLAP):
    # segment y_filtered into consecutive non-overlapping intervals
    segments = [TS[i:i+TIMESTEP] for i in range(0, len(TS)-TIMESTEP+1, TIMESTEP-OVERLAP)]
    print(np.array(segments).shape)

    # construct a visibility graph for each segment
    graphs = [construct_visibilty_graph(segment) for segment in segments]
    print(np.array(graphs).shape)

    # compute feature vectors for each graph using degree centrality
    feature_vectors = [np.array(list(nx.degree_centrality(graph).values())) for graph in graphs]
    print(np.array(feature_vectors).shape)

    # define distance matrix D for each segment using Euclidean distance
    distance_matrices = []
    for i in range(len(feature_vectors)):
        D = np.zeros((len(feature_vectors), len(feature_vectors)))
        for j in range(len(feature_vectors)):
            D[i][j] = np.linalg.norm(feature_vectors[i]-feature_vectors[j])
        distance_matrices.append(D)

    # compute global distance matrix by averaging distances across all segments
    global_distance_matrix = np.mean(distance_matrices, axis=0)
    print(np.array(global_distance_matrix).shape)

    # normalize global distance matrix between 0 and 1
    normalized_distance_matrix = 1 - (global_distance_matrix - np.min(global_distance_matrix)) / (np.max(global_distance_matrix) - np.min(global_distance_matrix))
    print(np.array(normalized_distance_matrix).shape)

    # construct weighted graph C using normalized distance matrix as adjacency matrix
    # threshold = 0.0  # set a threshold to remove weak edges
    # C = nx.from_numpy_array(normalized_distance_matrix * (normalized_distance_matrix >= threshold))

    C = nx.from_numpy_array(normalized_distance_matrix)

    # Set edge weights to similarity values
    for u, v, d in C.edges(data=True):
        d['weight'] = normalized_distance_matrix[u][v]
    
    # Remove self-loops
    C.remove_edges_from(nx.selfloop_edges(C))

    return C

def generate_partitions(weightedGraph):
    return community_louvain.best_partition(weightedGraph, resolution=(1.02))
    
def majority_partition(partition, TS, TIMESTEP, OVERLAP):
    """
    Generate majority partition from original timeseries and generated partitions

    Args:
        partition (list[int]): A list of integers that represent the partition to which each segment in `TS` belongs.
        TS (list[float]): A time series represented as a list of floats.
        TIMESTEP (int): The length of each time step.
        OVERLAP (int): The amount of overlap between consecutive time steps.

    Returns:
        dict[int, int]: A dictionary where the keys are the time steps of `TS`, and the values are the majority partition for each time step.
    """
    TRUNCATED = len(TS) % TIMESTEP
    TSFAKE = [*range(len(TS))]
    SEGMENTSFAKE = [TSFAKE[i:i+TIMESTEP] for i in range(0, len(TS)-TIMESTEP+1, TIMESTEP-OVERLAP)] #(from 0 to last possible timestep, in steps of len of timestep (remove overlap))
    occuranceTable = {i: [segmentNum for segmentNum in range(len(SEGMENTSFAKE)) if i in SEGMENTSFAKE[segmentNum]] for i in range(len(TS) - TRUNCATED) if any(i in seg for seg in SEGMENTSFAKE)}

    majority_partition = {}
    # iterate over time steps
    for i in range(len(TS) - TRUNCATED):
        partitions = {} # dictionary to store frequency of partitions in current time step
        
        # iterate over segments in current time step
        for segmentNum in occuranceTable[i]:
            p = partition[segmentNum] # get partition of current segment
            if p not in partitions:
                partitions[p] = 1
            else:
                partitions[p] += 1
            
        # assign majority partition for current timestep
        majority = max(partitions, key=partitions.get) 
        majority_partition[i] = majority

    return majority_partition

def contrasting_colors(n_colors):
    """
    Generates a dictionary of contrasting colors.
    """
    import colorsys
    colors = {}
    for i in range(n_colors): # generates highly contrasting colors by incrementing hue
        hue = i / float(n_colors)
        saturation = 0.9
        value = 0.9
        (r, g, b) = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = (r, g, b)

    return colors