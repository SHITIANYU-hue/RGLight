#from flow.envs.base_env import Env
#class Lane(Env):
import traci 

class Edge():
    def __init__(self, edge_id):

        #1) INIT 
            #ID 
        self.edge_id= edge_id
        self.next_edges = set()
        self.prev_edges = set()
