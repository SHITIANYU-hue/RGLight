import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
import numpy as np 
import dgl
from functools import partial
from torch.autograd import Variable
import copy
from utils.mdn import *
from IPython.display import clear_output
import glob
import atexit
import time
import traceback


# FUNCTIONS TO CREATE LOCAL GRAPHS 
def get_adjacency_matrix(lane_connection):
    edges = []
    for i in lane_connection.getIDList():
        if "center" not in i:
            for idx,j in enumerate(lane_connection.getLinks(i)):
                try:
                    edges.append((i,j[0],1))
                except:
                    pass
    return edges

def get_inb_adj_vector(graph, lane_id):
    return inb_adj_vector
def get_outb_adj_vector(graph, lane_id):
    return outb_adj_vector
def get_discount_vector_tl(graph, tl_id):     # LEAVE A CHOICE FOR THE MEASURE OF DISTANCE 
    return distance_vector, discount_vector
def get_distance_vector_lane(graph,lane_id):
    return distance_vector, discount_vector


# USE FUNCTION IN "FROMHERE" file
# NOT REALLY NESCESSARY... CAN USE TRACI FUNCTION INSTEAD
def get_controlled_lanes(graph, tl_id):
    return inbound_lanes, outbound_lanes, connections

#Create own reward function or use existing one
def compute_reward(veh_data):
    return reward 



from collections import defaultdict

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

    
def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_nodes = []
        next_node = shortest_paths[current_node][0]
        
        current_node = next_node
    # Reverse path
    path = path[::-1]
#     print("path", path, "len", len(path)-1)
 #   print(path, len(path))
    return path, len(path)-1, shortest_paths





# FILTER FUNCTIONS

def filt(nodes, identifier): 
    mask = ''
    for idx,i in enumerate(identifier):
        mask+='(nodes.data["node_type"]==' + str(i) +')'
        if idx != len(identifier)-1:
            mask+= '|'
    return (eval(mask))#.squeeze(1)
"""

def is_lane(nodes): return (nodes.data['node_type'] == 2)#.squeeze(1)
def is_veh(nodes): return (nodes.data['node_type'] == 3)#.squeeze(1)
def is_edge(nodes): return (nodes.data['node_type'] == 4)#.squeeze(1)
def is_connection(nodes): return (nodes.data['node_type'] == 5)#.squeeze(1)
def is_phase(nodes): return (nodes.data['node_type'] == 6)#.squeeze(1)

"""



"""
g.filter_nodes(is_tl)

g.filter_nodes(is_lane)

g.filter_nodes(is_connection)

g.filter_nodes(is_phase)

g.filter_nodes(is_veh)
"""
    
