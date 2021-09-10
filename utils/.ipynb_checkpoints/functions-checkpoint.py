from itertools import compress, product
import collections
import numpy as np
import matplotlib.pyplot as plt
import random
import os, sys
import cv2
from collections import deque
from os.path import isfile, join
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")
import sumolib


plt.style.use(['dark_background'])



color = True



def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)



def get_viz():
    return colo 

class Memory():
    def __init__(self, manager = False, max_size = 100000):
        if manager:
            self.manager = manager
            self.buffer = self.manager.list()
        else:
            self.buffer = []
        self.max_size = max_size
        
    def add(self, experience):
        if len(self.buffer)>=self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]
    
    
class Memory_weighted():
    def __init__(self, manager, max_size):
        self.manager = manager
        self.buffer = self.manager.list()
        self.actions_taken = self.manager.list()
        self.max_size = max_size
        
    def add(self, experience, action):
        if len(self.buffer)>=self.max_size:
            self.buffer.pop(0)
            self.actions_taken.pop(0)
        self.buffer.append(experience)
        self.actions_taken.append(action)
    
    def sample(self, batch_size):
        self.counter = collections.Counter(self.actions_taken)
        probs = [self.counter.get(i) for i in self.actions_taken]
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False,
                                p = probs)
        
        return [self.buffer[i] for i in index]


def combinations(items):
    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )
    
def gen_random_route(max_num_veh_per_step, step, p):
    for i in range(max_num_veh_per_step):
        if random.uniform(0, 1) < p:
            edge = random.choice(edges)
            traci.route.add("trip_"+str(step)+"_"+str(i), [edge, random.choice(traci.lane.getLinks(edge))])
            traci.vehicle.add("newVeh_"+str(step)+"_"+str(i), "trip")
            
import optparse
import traci
def get_edges():
    edges = traci.edge.getIDList()
    return edges

def get_traffic_lights():
    traffic_lights_lanes = {}
    traffic_lights_action_counts = {}
  #  traffic_lights_phases_durations = {}
    net = sumolib.net.readNet("Mtl/Mtl.net.xml")
    for traffic_light in net._tlss:     
    #    traffic_lights_phases_durations[traffic_light.getID()]={}
        traffic_lights_lanes[traffic_light.getID()]={}  
        action_count = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light.getID())[0]._phases)
        traffic_lights_action_counts[traffic_light.getID()] = action_count
    #    for phase in action_count:
   #         traffic_lights_phases_durations[traffic_light.getID()][phase] = 0
     #   traffic_lights_phases_durations[traffic_light_getID()]
        for connection in traffic_light._connections:
            traffic_lights_lanes[traffic_light.getID()][connection[0].getID()]=[]
            traffic_lights_lanes[traffic_light.getID()][connection[1].getID()]=[]
    return traffic_lights_lanes, traffic_lights_action_counts#, traffic_lights_phases_durations
        
def get_traffic_lights2():
    traffic_lights = {}
    tl_networks = {}
    net = sumolib.net.readNet("Mtl/Mtl.osm.net.xml")
    for traffic_light in net._tlss:
        traffic_lights[traffic_light.getID()]={}   
        for connection in traffic_light._connections:
            traffic_lights[traffic_light.getID()][connection[0].getID()]=[]
            traffic_lights[traffic_light.getID()][connection[1].getID()]=[]
            
        tl_networks[traffic_light] = NN.NN(len(traffic_lights[traffic_light.getID()])*len())
    return traffic_lights, networks
    #         print("inward connection :", connection[0].getID(), "outward connection:", connection[1].getID())
        

def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    with open("cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>
        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>
def mov_avg(ensemble, nb_episodes_moving_avg = None):
    N = len(ensemble)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = np.asarray(ensemble[max(0, t+1-nb_episodes_moving_avg):(t+1)]).mean()
    return moving_avg



def plot(name, write, ensemble1, ensemble2 = None, ensemble3 = None, ensemble4 = None, nb_episodes_moving_avg = None):
    if ensemble2 != None:
        plt.plot(mov_avg(ensemble1, nb_episodes_moving_avg), color = 'w', label = "agent")    
        plt.plot(mov_avg(ensemble2, nb_episodes_moving_avg), color = 'b', label = "osm_policy")
        if ensemble3 != None:
            #plt.ylim(-10000, -4000)            
            plt.plot(mov_avg(ensemble3, nb_episodes_moving_avg), color = 'c', label = "motion_based_policy")    
            if ensemble4 != None:
                plt.plot(mov_avg(ensemble4, nb_episodes_moving_avg), color = 'r', label = "agent_policy")
                
        plt.legend(loc='lower left')
        plt.xlabel("Number of tested episodes", color = 'w')
        plt.ylabel("Reward", color = 'w')
    else:
        plt.plot(mov_avg(ensemble1, nb_episodes_moving_avg), color = 'w')
        
    plt.title(str(name), color='w')
    if write:
        plt.savefig("Plots/"+name+'.png')            
    plt.show()
    plt.clf()

def plot_last10(name, write, ensemble1, ensemble2 = None, ensemble3 = None, ensemble4 = None, nb_episodes_moving_avg = None):
    if ensemble2 != None:
        plt.plot(mov_avg(ensemble1[max(0,len(ensemble1)-nb_episodes_moving_avg):], nb_episodes_moving_avg), color = 'w', label = "agent")    
        plt.plot(mov_avg(ensemble2[max(0,len(ensemble1)-nb_episodes_moving_avg):], nb_episodes_moving_avg), color = 'b', label = "osm_policy")
        if ensemble3 != None:
            #plt.ylim(-10000, 0)
            plt.plot(mov_avg(ensemble3[max(0,len(ensemble1)-nb_episodes_moving_avg):], nb_episodes_moving_avg), color = 'c', label = "motion_based_policy")    
            if ensemble4 != None:
                plt.plot(mov_avg(ensemble4[max(0,len(ensemble1)-nb_episodes_moving_avg):], nb_episodes_moving_avg), color = 'r', label = "agent_policy")
                
        plt.legend(loc='lower left')
        plt.xlabel("Number of tested episodes", color = 'w')
        plt.ylabel("Reward", color = 'w')
    else:
        plt.plot(mov_avg(ensemble1[max(0,len(ensemble1)-nb_episodes_moving_avg):], nb_episodes_moving_avg), color = 'w')
        
    plt.title(str(name), color='w')
    if write:
        plt.savefig("Plots/"+name+'.png')            
    plt.show()
    plt.clf()

def save_plot(name, write, ensemble1, ensemble2 = None, ensemble3 = None, ensemble4 = None, nb_episodes_moving_avg = None):
    if ensemble2 != None:
        plt.plot(mov_avg(ensemble1, nb_episodes_moving_avg), color = 'w', label = "agent")    
        plt.plot(mov_avg(ensemble2, nb_episodes_moving_avg), color = 'b', label = "osm_policy")
        if ensemble3 != None:
            #plt.ylim(-10000, 0)            
            plt.plot(mov_avg(ensemble3, nb_episodes_moving_avg), color = 'c', label = "motion_based_policy")    
            if ensemble4 != None:
                plt.plot(mov_avg(ensemble4, nb_episodes_moving_avg), color = 'r', label = "agent_policy")

        plt.legend(loc='lower left')
        plt.xlabel("Number of tested episodes", color = 'w')
        plt.ylabel("Reward", color = 'w')
    else:
        plt.plot(mov_avg(ensemble1, nb_episodes_moving_avg), color = 'w')

    plt.title(str(name), color='w')
    if write:
        plt.savefig("Plots/"+name+'.png')            
    plt.clf()

    
def plot2(name, write, ensemble1, ensemble2 = None, ensemble3 = None, ensemble4 = None, nb_episodes_moving_avg = None):
    if ensemble2 != None:
        plt.plot(ensemble1, color = 'w', label = "agent")
        plt.plot(ensemble2, color = 'b', label = "osm_policy")
        if ensemble3 != None:
            #plt.ylim(-10000, 0)
            plt.plot(ensemble3, color = 'c', label = "motion_based_policy")
            if ensemble4 != None:
                plt.plot(ensemble4, color = 'r', label = "agent_policy")
                
        plt.legend(loc='lower left')
        plt.xlabel("Number of tested episodes", color = 'w')
        plt.ylabel("Reward", color = 'w')
    else:
        plt.plot(ensemble1, color = 'w')
        
    plt.title(str(name), color='w')
    if write:
        plt.savefig("Plots/"+name+'.png')            
    plt.show()
    plt.clf()
    
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))
    files.sort()
 
    for i in range(len(files)):
        print(files[i])
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 




def main():
    pathIn= '/Videos/Captures'
    pathOut = str('/Videos/Videos/video_episode_'+str(episode+1)+'.avi')
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
 
if __name__=="__main__":
    main()