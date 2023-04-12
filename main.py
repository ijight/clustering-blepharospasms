import scipy
import numpy as np
import matplotlib.pyplot as plt
import os 
import networkx as nx
import igraph as ig
from ts2vg import NaturalVG
from scipy.spatial.distance import euclidean
from networkx import to_numpy_array

singlechannel_timeseries = []
datadir = "Singlechannel\PreviousData_v2\\"
for each in os.listdir(datadir):
    singlechannel_timeseries.append(list(scipy.io.loadmat(datadir+each)["X"][0]))

