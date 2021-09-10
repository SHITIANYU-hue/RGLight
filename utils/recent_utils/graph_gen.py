def setup(self):
    setup_variables()
    self.env.number_of_vehicles = 0
    setup_classes_and_networks()
    setup_paths()
    
def setup_variables(self):
    # EXTEND THE DICTIONNARIES OF NODE VARIABLES WITH VARIABLES REPRESENTING AN ARBITRARY NUMBER OF ENTITIES 
    # ARBITRARY NUMBER OF VEHICLES REPRESENTED ON A LANE NODE 
    if self.env.env_params.additional_params['veh_state'] and self.env.env_params.additional_params['lane_node_state']:
        for veh_idx in range(self.env.env_params.additional_params['num_observed']):
            for var_name, var_dim in self.env.env_params.additional_params['lane_per_veh_vars'].items():
                self.env.env_params.additional_params['lane_vars'][str(str(veh_idx) + "_" + var_name)] = var_dim # SPEED

    # ARBITRARY NUMBER OF NEXT_PHASES REPRESENTED ON A CONNECTION NODE 
    if self.env.env_params.additional_params['phase_state'] :
        for phase_idx in range(self.env.env_params.additional_params['num_observed_next_phases']):
            for var_name, var_dim in self.env.env_params.additional_params['connection_per_phase_vars'].items():
                self.env.env_params.additional_params['connection_vars'][str(str(phase_idx) + "_" + var_name)] = var_dim # SPEED

    
def setup_classes_and_networks(self):
    
    
    #INITIALIZE OBJECTS 
    self.env.Nodes_connections = collections.OrderedDict()
    self.env.Agent = Agent
    self.env.Agent.Policy_Type = self.env.env_params.additional_params['Policy_Type']
    self.env.Lane = Lane
    self.env.Lane.veh_state = self.env.env_params.additional_params['veh_state']
    self.env.Lane.n_observed = self.env.env_params.additional_params['num_observed']
    self.env.Lane.stopped_delay_threshold = self.env.env_params.additional_params['stopped_delay_threshold']
    self.env.shortest_paths = {}
    self.env.full_lane_connections=[]
    self.env.Agents = collections.OrderedDict()
    self.env.center = {}
    self.env.lanes = []
    
    
    # CREATE AGENTS (TRAFFIC LIGHTS) INSTANCES AND RELEVANT CONNECTIONS OBJECTS
    for tl_id in self.env.traci_connection.trafficlight.getIDList():
        self.env.Agents[tl_id]=Agent(tl_id)
        self.env.Agents[tl_id].phases_defs = [i._phaseDef for i in self.env.traci_connection.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]._phases]
        self.env.Agents[tl_id].n_phases = len(self.env.traci_connection.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]._phases)
        self.env.Agents[tl_id].distance_vector = []
        self.env.Agents[tl_id].discount_vector = []
        self.env.Agents[tl_id].unordered_connections_trio = []
        self.env.Agents[tl_id].distance_dic = collections.OrderedDict()
        self.env.Nodes_connections[tl_id] = []

        for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id):
            self.env.Agents[tl_id].unordered_connections_trio.append(link)

        for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id) :
            if link[0][0] not in [x[0] for x in self.env.Nodes_connections[tl_id]]:
                self.env.Nodes_connections[tl_id].append((link[0][0], "inbound"))
            if link[0][1] not in [x[0] for x in self.env.Nodes_connections[tl_id]]:
                self.env.Nodes_connections[tl_id].append((link[0][1], "outbound"))
            self.env.Agents[tl_id].inb_lanes.append(link[0][0])
            self.env.Agents[tl_id].outb_lanes.append(link[0][1])
            name = str(str(link[0][0])+"_"+str(link[0][1]))
            self.env.center[name]=link[0][2]
            self.env.lanes.extend([link[0][0], link[0][1]])
            self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][0])
            self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][1])
            self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][2])
            self.env.full_lane_connections.append([link[0][0],link[0][2],1])
            self.env.full_lane_connections.append([link[0][2],link[0][1],1])
            self.env.Agents[tl_id].connections_trio.append(link)

        self.env.Agents[tl_id].inb_lanes = set(self.env.Agents[tl_id].inb_lanes)
        self.env.Agents[tl_id].outb_lanes = set(self.env.Agents[tl_id].outb_lanes)
        self.env.Agents[tl_id].complete_controlled_lanes = set(self.env.Agents[tl_id].complete_controlled_lanes)


        
    # GET SETS OF CENTRAL LANES (INTERIOR PART OF AN INTERSECTION) AND NORMAL LANES (LINKING INTERSECTIONS)
    self.env.central_lanes = set(self.env.center.values())
    self.env.lanes = set(self.env.lanes)
    
    # DICTIONNARY DEFINING IDENTIFIER OF EVERY TYPE OF NODE IN THE DIFFERENT NETWORKS
    self.env.nodes_types = {1: 'tl', 2: 'lane', 3: 'veh', 4: 'edge', 5: 'connection', 6: 'phase'}

    # FOR PARRALLEL COMPUTATION TO BE PERFORMABLE, NODE STATES HAVE TO BE OF THE SAME SIZE. WE THEREFORE DEFINE THE NODE_STATE_SIZE AS THE MAXIMUM STATE SIZE AMONG EVERY TYPE OF NODES
    self.env.node_state_size = max(sum(self.env.env_params.additional_params["tl_vars"].values()), sum(self.env.env_params.additional_params["lane_vars"].values()), sum(self.env.env_params.additional_params["edge_vars"].values()), sum(self.env.env_params.additional_params["connection_vars"].values()), sum(self.env.env_params.additional_params["phase_vars"].values()))

    if self.env.env_params.additional_params['veh_as_nodes']:
        self.env.node_state_size = max(self.env.node_state_size, sum(self.env.env_params.additional_params["veh_vars"].values()))


    # WE GENERATE ALL THE NETWORKS INCLUDED IN ('generated_graphs') FOR DEEP GRAPH LIBRARY 
    generate_networks()       
        

    print("original graphs", self.env.original_graphs)


    
    # WE GET PREVIOUS AND FOLLOWING LANES FOR EVERY LANE (WILL BE USED TO IDENTIFY STARTING LANES AND ENDING LANES FOR VALID ROUTES)
    for tl_id in self.env.traci_connection.trafficlight.getIDList():                
        for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id) :


            if tl_id not in [x[0] for x in self.env.Nodes_connections[link[0][0]]]:                    
                self.env.Nodes_connections[link[0][0]].append((tl_id, "outbound"))

            if tl_id not in [x[0] for x in self.env.Nodes_connections[link[0][1]]]:     
                self.env.Nodes_connections[link[0][1]].append((tl_id, "inbound"))


            self.env.Lanes[link[0][0]].next_tl = tl_id
            self.env.Lanes[link[0][1]].previous_tl = tl_id                






    #self.env.phantom_connections = []
    
    # COMPUTE DISTANCES (FOR REWARD COMPUTATION) AND SHORTEST PATHS (TO CREATE SENSIBLE ROUTES FOR VEHICLES)
    for idx1, tl_id in enumerate(self.env.traci_connection.trafficlight.getIDList()):

        for idx2, lane_id in enumerate(self.env.valid_lanes):

            for idx3, controlled_lane in enumerate(self.env.Agents[tl_id].complete_controlled_lanes):

                # NEW OPTIONNAL
                if lane_id in self.env.lanes and controlled_lane in self.env.lanes:


                    #try:
                    if controlled_lane not in self.env.Lanes[lane_id].distance_dic:
                        shortest_paths = [p for p in nx.all_shortest_paths(self.env.lane_graph,source=lane_id,target=controlled_lane)]
                        distance = len(shortest_paths[0]) -1
                        #NEW CHANGE 

                        self.env.Lanes[lane_id].distance_dic[controlled_lane]= distance
                    else:
                        distance = self.env.Lanes[lane_id].distance_dic[controlled_lane]


                    if lane_id not in self.env.Agents[tl_id].distance_dic:
                        self.env.Agents[tl_id].distance_dic[lane_id]=distance

                    elif distance < self.env.Agents[tl_id].distance_dic[lane_id]:
                        self.env.Agents[tl_id].distance_dic[lane_id] = distance


                    #except:
                        #self.env.phantom_connections.append([lane_id,controlled_lane])
                        #pass


                if lane_id in self.env.lanes and controlled_lane in self.env.lanes and not self.env.Lanes[lane_id].inb_adj_lanes and not self.env.Lanes[controlled_lane].outb_adj_lanes and lane_id != str('-' + controlled_lane) and controlled_lane != str('-' + lane_id):

                    trip_name = str("route_" + lane_id + "_" + controlled_lane)     
                    if trip_name not in self.env.trip_names:  
                        self.env.trip_names.append(trip_name)
                        shortest_paths = [p for p in nx.all_shortest_paths(self.env.edge_graph,source=lane_id[:-2],target=controlled_lane[:-2])]
                        distance = len(shortest_paths[0]) -1

                        self.env.shortest_paths[trip_name] = shortest_paths

                        self.env.entering_edges.add(lane_id)
                        self.env.leaving_edges.add(controlled_lane)                             


                        
                        
        # WE CREATE A DISCOUNT VECTOR (BASED ON DISTANCES AND DISTANCE GAMMA) THAT WILL ENABLE EFFICIENT COMPUTATION OF THE REWARD FOR EVERY AGENT BASED ON NETWORK BASED DISTANCE 
        self.env.Agents[tl_id].distance_vector = np.asarray(list(collections.OrderedDict(sorted(self.env.Agents[tl_id].distance_dic.items(), key=lambda t: t[:-2])).values()))

        self.env.Agents[tl_id].discount_vector = self.env.env_params.additional_params["distance_gamma"]**self.env.Agents[tl_id].distance_vector

    for lane_id in self.env.Lanes:

        self.env.Nodes_connections[lane_id] = sorted(self.env.Nodes_connections[lane_id], key = lambda t: (t))
        self.env.Lanes[lane_id].distance_vector = np.asarray(list(collections.OrderedDict(sorted(self.env.Lanes[lane_id].distance_dic.items(), key=lambda t: t[:-2])).values()))
        self.env.Lanes[lane_id].discount_vector = self.env.env_params.additional_params["distance_gamma"]**self.env.Lanes[lane_id].distance_vector


    self.env.Nodes_connections = collections.OrderedDict(sorted(self.env.Nodes_connections.items(), key = lambda t: (-int(t[0].count('_')),t[0])))



def generate_networks(self):


    # TL NETWORK (DIRECTED)   ( )
    if 'tl_graph' in self.env.env_params.additional_params['generated_graphs'] :



        self.env.tl_graph_dgl = dgl.DGLGraph()
        self.env.tl_graph_dgl.nodes_types = collections.OrderedDict()
        self.env.tl_graph_dgl.nodes_types['tl'] = 1 
        self.env.tl_graph_dgl.adresses_in_graph = collections.OrderedDict()
        self.env.tl_graph_dgl.norms = collections.OrderedDict()
        self.env.tl_graph_dgl.adresses_in_sumo = collections.OrderedDict()
        src = []
        dst = []
        tp = []
        norm = []
        node_type = []
        counter = 0
        for tl_id in self.env.Agents:    

            #CREATE NODE 
            if tl_id not in self.env.tl_graph_dgl.adresses_in_graph:
                self.env.tl_graph_dgl.adresses_in_graph[tl_id] = counter
                self.env.tl_graph_dgl.norms[tl_id] = [0]*2
                self.env.tl_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                node_type.append(1)
                counter +=1 

                #ADD SELF LOOP WITH TYPE AT THE END 
                src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                self.env.tl_graph_dgl.norms[tl_id][-1] += 1     
                tp.append(len(self.env.tl_graph_dgl.norms[tl_id])-1)                         



            for tl_id2 in self.env.Agents:
                if tl_id2 not in self.env.tl_graph_dgl.adresses_in_graph:
                    self.env.tl_graph_dgl.adresses_in_graph[tl_id2] = counter
                    self.env.tl_graph_dgl.norms[tl_id2] = [0]*2
                    self.env.tl_graph_dgl.adresses_in_sumo[str(counter)] = tl_id2
                    node_type.append(1)
                    counter +=1

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                    dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                    self.env.tl_graph_dgl.norms[tl_id2][-1] += 1     
                    tp.append(len(self.env.tl_graph_dgl.norms[tl_id2])-1)       

                if list(set(self.env.Agents[tl_id].complete_controlled_lanes).intersection(self.env.Agents[tl_id2].complete_controlled_lanes)) and tl_id != tl_id2: # CHECK IF LINK EXISTS
                    src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                    self.env.tl_graph_dgl.norms[tl_id2][0] += 1     
                    tp.append(0)

        for destination, t in zip(dst,tp):
            norm.append([(1/self.env.tl_graph_dgl.norms[self.env.tl_graph_dgl.adresses_in_sumo[str(destination)]][t])])


        num_nodes = counter

        self.env.tl_graph_dgl.add_nodes(num_nodes)
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        edge_type = torch.LongTensor(tp)
        edge_norm = torch.FloatTensor(norm).squeeze()
        node_type = torch.LongTensor(node_type)

        self.env.tl_graph_dgl.add_edges(src,dst)                        
        self.env.tl_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                        
        self.env.tl_graph_dgl.ndata.update({'node_type' : node_type})        


    self.env.lane_graph = nx.Graph()
    self.env.edge_graph = nx.Graph()

    if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :            

        self.env.lane_graph_dgl = dgl.DGLGraph()
        self.env.lane_graph_dgl.nodes_types = collections.OrderedDict()
        self.env.lane_graph_dgl.nodes_types['lane'] = 1
        if self.env.env_params.additional_params['veh_as_nodes'] :
            self.env.lane_graph_dgl.nodes_types['veh'] = 2
        node_type = []
        self.env.lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
        self.env.lane_graph_dgl.norms = collections.OrderedDict()
        self.env.lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

    self.env.full_graph = Graph() 

    for lane_connection in self.env.full_lane_connections:
        self.env.full_graph.add_edge(*lane_connection)    



    self.env.valid_lanes = (collections.OrderedDict(sorted(self.env.full_graph.edges.items(), key=lambda t: t[0]))).keys()
    self.env.valid_lanes = list(self.env.valid_lanes)

    self.env.Edges = collections.OrderedDict()
    self.env.Lanes = collections.OrderedDict()      
    self.env.lane_connections = []
    self.env.edge_connections = []

    for edge_id in self.env.traci_connection.edge.getIDList():
        self.env.Edges[edge_id]=Edge(edge_id)
        self.env.Edges[edge_id].next_edges = set()

    for lane_id in self.env.lanes:
            self.env.Lanes[lane_id]=Lane(lane_id)
            self.env.traci_connection.lane.setMaxSpeed(lane_id, self.env.env_params.additional_params['Max_Speed'])    
            self.env.Lanes[lane_id].max_speed = self.env.traci_connection.lane.getMaxSpeed(lane_id)
            self.env.Lanes[lane_id].length = self.env.traci_connection.lane.getLength(lane_id)
            self.env.Lanes[lane_id].std_length = (self.env.traci_connection.lane.getLength(lane_id)-self.env.env_params.additional_params['min_lane_length'])/(self.env.env_params.additional_params['max_lane_length']-self.env.env_params.additional_params['min_lane_length'])
            self.env.Lanes[lane_id].distance_dic = collections.OrderedDict()

    counter = 0


    if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
        src = []
        dst = []
        tp = []
        norm = []
    for idx, lane_id in enumerate(self.env.lanes):

            if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
                if lane_id not in self.env.lane_graph_dgl.adresses_in_graph:
                    self.env.lane_graph_dgl.adresses_in_graph[lane_id] = counter
                    self.env.lane_graph_dgl.norms[lane_id] = [0]*3
                    self.env.lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                    node_type.append(1)
                    counter += 1 

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                    self.env.lane_graph_dgl.norms[lane_id][-1] += 1     
                    tp.append(len(self.env.lane_graph_dgl.norms[lane_id])-1)  

            self.env.Nodes_connections[lane_id] = []
            edge_id = self.env.traci_connection.lane.getEdgeID(lane_id)

            try:
                self.env.Edges[edge_id]
            except:
                self.env.Edges[edge_id] = Edges(edge_id)



            for idx,info_connection in enumerate(list(self.env.traci_connection.lane.getLinks(lane_id))):      

                if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
                    if info_connection[0] not in self.env.lane_graph_dgl.adresses_in_graph:  
                        self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]] = counter
                        self.env.lane_graph_dgl.norms[info_connection[0]] = [0]*3
                        self.env.lane_graph_dgl.adresses_in_sumo[str(counter)] = info_connection[0]
                        node_type.append(2)
                        counter += 1 

                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                        dst.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                        self.env.lane_graph_dgl.norms[info_connection[0]][-1] += 1     
                        tp.append(len(self.env.lane_graph_dgl.norms[info_connection[0]])-1)                         

                    # vehicle flow
                    src.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                    self.env.lane_graph_dgl.norms[info_connection[0]][0] +=1
                    tp.append(0)
                    # reverse flow
                    src.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                    dst.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                    self.env.lane_graph_dgl.norms[lane_id][1] +=1
                    tp.append(1)

                self.env.lane_graph.add_node(lane_id)
                self.env.lane_graph.add_node(info_connection[0])                        
                self.env.lane_graph.add_edge(lane_id,info_connection[0])
                self.env.edge_graph.add_node(lane_id[:-2])
                self.env.edge_graph.add_node(info_connection[0][:-2])                        
                self.env.edge_graph.add_edge(lane_id[:-2],info_connection[0][:-2])                        

                self.env.edge_connections.append((lane_id[:-2],info_connection[0][:-2],1))                           
                self.env.lane_connections.append((lane_id,info_connection[0],1))   
                self.env.Lanes[lane_id].outb_adj_lanes.append(info_connection[0])
                next_edge = self.env.traci_connection.lane.getEdgeID(info_connection[0])
                self.env.Edges[edge_id].next_edges.add(next_edge)
                self.env.Edges[next_edge].prev_edges.add(edge_id)
                self.env.Lanes[info_connection[0]].inb_adj_lanes.append(lane_id)   




    if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
        for destination, t in zip(dst,tp):
            norm.append([(1/self.env.lane_graph_dgl.norms[self.env.lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



        num_nodes = counter

        self.env.lane_graph_dgl.add_nodes(num_nodes)
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        edge_type = torch.LongTensor(tp)
        edge_norm = torch.FloatTensor(norm).squeeze()
        node_type = torch.LongTensor(node_type)


        self.env.lane_graph_dgl.add_edges(src,dst)                        
        self.env.lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                        
        self.env.lane_graph_dgl.ndata.update({'node_type' : node_type})                            



    self.env.lane_connections=list(set(tuple(i) for i in self.env.lane_connections))                    
    self.env.routes = []
    self.env.trip_names = []
    self.env.entering_edges = set()
    self.env.leaving_edges = set()
    self.env.entering_edges_probs = []
    self.env.leaving_edges_probs = []

    if 'tl_lane_graph' in self.env.env_params.additional_params['generated_graphs'] :

        self.env.tl_lane_graph_dgl = dgl.DGLGraph()
        self.env.tl_lane_graph_dgl.nodes_types = collections.OrderedDict()
        self.env.tl_lane_graph_dgl.nodes_types['tl'] = 1
        self.env.tl_lane_graph_dgl.nodes_types['lane'] = 2
        if self.env.env_params.additional_params['veh_as_nodes'] :
            self.env.tl_lane_graph_dgl.nodes_types['veh'] = 3
        self.env.tl_lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
        self.env.tl_lane_graph_dgl.norms = collections.OrderedDict()
        self.env.tl_lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

        counter = 0
        src = []
        dst = []
        tp = []
        norm = []
        node_type = []
        for tl_id in self.env.Agents:
            if tl_id not in self.env.tl_lane_graph_dgl.adresses_in_graph: 
                self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id] = counter
                self.env.tl_lane_graph_dgl.norms[tl_id] = [0]*6
                self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                node_type.append(1)
                counter+=1

                #ADD SELF LOOP WITH TYPE AT THE END 
                src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                self.env.tl_lane_graph_dgl.norms[tl_id][-1] += 1     
                tp.append(len(self.env.tl_lane_graph_dgl.norms[tl_id])-1)                              


            for lane_id in self.env.Agents[tl_id].inb_lanes:
                if lane_id not in self.env.tl_lane_graph_dgl.adresses_in_graph:
                    self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                    self.env.tl_lane_graph_dgl.norms[lane_id] = [0]*6
                    self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                    node_type.append(2)
                    counter += 1 

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    self.env.tl_lane_graph_dgl.norms[lane_id][-2] += 1     
                    tp.append(len(self.env.tl_lane_graph_dgl.norms[lane_id])-2)                              


                # vehicle flow
                src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                # reverse flow
                src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                self.env.tl_lane_graph_dgl.norms[lane_id][0] +=1
                tp.append(0)
                self.env.tl_lane_graph_dgl.norms[tl_id][1] +=1                        
                tp.append(1)                    


            for lane_id in self.env.Agents[tl_id].outb_lanes:
                if lane_id not in self.env.tl_lane_graph_dgl.adresses_in_graph:
                    self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                    self.env.tl_lane_graph_dgl.norms[lane_id] = [0]*6
                    self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                    node_type.append(2)
                    counter += 1 

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    self.env.tl_lane_graph_dgl.norms[lane_id][-2] += 1     
                    tp.append(len(self.env.tl_lane_graph_dgl.norms[lane_id])-2)                              


                # vehicle flow
                src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                # reverse flow
                src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                self.env.tl_lane_graph_dgl.norms[lane_id][2] +=1
                tp.append(2)
                self.env.tl_lane_graph_dgl.norms[tl_id][3] +=1                        
                tp.append(3)                               


        for destination, t in zip(dst,tp):
            norm.append([(1/self.env.tl_lane_graph_dgl.norms[self.env.tl_lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



        num_nodes = counter


        self.env.tl_lane_graph_dgl.add_nodes(num_nodes)
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        edge_type = torch.LongTensor(tp)
        edge_norm = torch.FloatTensor(norm).squeeze()
        node_type = torch.LongTensor(node_type)

        self.env.tl_lane_graph_dgl.add_edges(src,dst)                        
        self.env.tl_lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                              
        self.env.tl_lane_graph_dgl.ndata.update({'node_type' : node_type})                            



    # NODE TYPES :  (TL : 1) (CONNECTION :5) (LANE :2)
    if 'tl_connection_lane_graph' in self.env.env_params.additional_params['generated_graphs']: 
        self.env.tl_connection_lane_graph_dgl = dgl.DGLGraph()
        self.env.tl_connection_lane_graph_dgl.nodes_types = collections.OrderedDict()
        self.env.tl_connection_lane_graph_dgl.nodes_types['tl'] = 1
        self.env.tl_connection_lane_graph_dgl.nodes_types['lane'] = 2
        self.env.tl_connection_lane_graph_dgl.nodes_types['connection'] = 3
        if self.env.env_params.additional_params['veh_as_nodes'] :
            self.env.tl_connection_lane_graph_dgl.nodes_types['veh'] = 4
        self.env.tl_connection_lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
        self.env.tl_connection_lane_graph_dgl.norms = collections.OrderedDict()
        self.env.tl_connection_lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

        counter = 0
        src = []
        dst = []
        tp = []
        norm = []
        node_type = []        


        # TLs
        for tl_id in self.env.Agents:

            # CREATE NODES
            if tl_id not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id] = counter                  
                self.env.tl_connection_lane_graph_dgl.norms[tl_id] = [0]*9
                self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                node_type.append(1)
                counter+=1


                #ADD SELF LOOP WITH TYPE AT THE END 
                src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                self.env.tl_connection_lane_graph_dgl.norms[tl_id][-1] += 1     
                tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[tl_id])-1)                         


            #LINKS/CONNECTIONS
            for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                link_name = str(tl_id+"_link_"+str(link_idx))

                if link_name not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                    # CREATE NODES 
                    self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name] = counter
                    self.env.tl_connection_lane_graph_dgl.norms[link_name] = [0]*9
                    self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = link_name
                    node_type.append(3)
                    counter+=1


                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                    self.env.tl_connection_lane_graph_dgl.norms[link_name][-2] += 1     
                    tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[link_name])-2) 



                # CREATE LINKS  
                # vehicle flow
                src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                # reverse flow
                src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])

                self.env.tl_connection_lane_graph_dgl.norms[link_name][0] +=1
                tp.append(0)
                self.env.tl_connection_lane_graph_dgl.norms[tl_id][1] +=1                        
                tp.append(1)   


                for dir_idx in range(2):                        
                    lane_id = link[0][dir_idx]

                    if lane_id not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                        # CREATE NODES 
                        self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                        self.env.tl_connection_lane_graph_dgl.norms[lane_id] = [0]*9
                        self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                        node_type.append(2)
                        counter+=1


                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                        self.env.tl_connection_lane_graph_dgl.norms[lane_id][-3] += 1     
                        tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[lane_id])-3)  



                    # CREATE LINKS 
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                    # reverse flow
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])


                    if dir_idx == 0:

                        self.env.tl_connection_lane_graph_dgl.norms[lane_id][2] +=1
                        tp.append(2)
                        self.env.tl_connection_lane_graph_dgl.norms[link_name][3] +=1                        
                        tp.append(3)     


                    elif dir_idx == 1:

                        self.env.tl_connection_lane_graph_dgl.norms[lane_id][4] +=1
                        tp.append(4)
                        self.env.tl_connection_lane_graph_dgl.norms[link_name][5] +=1                        
                        tp.append(5)     



        for destination, t in zip(dst,tp):
            norm.append([(1/self.env.tl_connection_lane_graph_dgl.norms[self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



        num_nodes = counter


        self.env.tl_connection_lane_graph_dgl.add_nodes(num_nodes)
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        edge_type = torch.LongTensor(tp)
        edge_norm = torch.FloatTensor(norm).squeeze()
        node_type = torch.LongTensor(node_type).squeeze()

        self.env.tl_connection_lane_graph_dgl.add_edges(src,dst)                        
        self.env.tl_connection_lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                        
        self.env.tl_connection_lane_graph_dgl.ndata.update({'node_type' : node_type})                            



















    if 'full_graph' in self.env.env_params.additional_params['generated_graphs'] :        


        #FULL GRAPH : IN CONSTRUCTION 


        self.env.full_graph_dgl = dgl.DGLGraph()
        self.env.full_graph_dgl.nodes_types = collections.OrderedDict()
        self.env.full_graph_dgl.nodes_types['tl'] = 1
        self.env.full_graph_dgl.nodes_types['lane'] = 2
        self.env.full_graph_dgl.nodes_types['edge'] = 3
        self.env.full_graph_dgl.nodes_types['connection'] = 4
        self.env.full_graph_dgl.nodes_types['phase'] = 5
        if self.env.env_params.additional_params['veh_as_nodes'] :
            self.env.full_graph_dgl.nodes_types['veh'] = 6
        self.env.full_graph_dgl.adresses_in_graph = collections.OrderedDict()
        self.env.full_graph_dgl.norms = collections.OrderedDict()
        self.env.full_graph_dgl.adresses_in_sumo = collections.OrderedDict()

        counter = 0
        src = []
        dst = []
        tp = []
        norm = []
        node_type = []        



        # TLs
        for tl_id in self.env.Agents:

            # CREATE NODES
            if tl_id not in self.env.full_graph_dgl.adresses_in_graph: 
                self.env.full_graph_dgl.adresses_in_graph[tl_id] = counter                  
                self.env.full_graph_dgl.norms[tl_id] = [0]*21
                self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                node_type.append(1)
                counter+=1


                #ADD SELF LOOP WITH TYPE AT THE END 
                src.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                self.env.full_graph_dgl.norms[tl_id][-1] += 1     
                tp.append(len(self.env.full_graph_dgl.norms[tl_id])-1)                         



            #PHASES 
            for phase_idx, phase in enumerate(self.env.Agents[tl_id].phases_defs):


                phase_name = str(tl_id+"_phase_"+str(phase_idx))
                if phase_name not in self.env.full_graph_dgl.adresses_in_graph: 
                    # CREATE NODES 
                    self.env.full_graph_dgl.adresses_in_graph[phase_name] = counter
                    self.env.full_graph_dgl.norms[phase_name] = [0]*21
                    self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = phase_name
                    node_type.append(5)
                    counter+=1


                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                    self.env.full_graph_dgl.norms[phase_name][-2] += 1     
                    tp.append(len(self.env.full_graph_dgl.norms[phase_name])-2)   





                # CREATE LINKS       
                # vehicle flow
                src.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                # reverse flow
                src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                dst.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])

                self.env.full_graph_dgl.norms[phase_name][0] +=1
                tp.append(0)
                self.env.full_graph_dgl.norms[tl_id][1] +=1                        
                tp.append(1)                    



                #LINKS/CONNECTIONS
                for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                    link_name = str(tl_id+"_link_"+str(link_idx))

                    if link_name not in self.env.full_graph_dgl.adresses_in_graph: 
                        # CREATE NODES 
                        self.env.full_graph_dgl.adresses_in_graph[link_name] = counter
                        self.env.full_graph_dgl.norms[link_name] = [0]*21
                        self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = link_name
                        node_type.append(4)
                        counter+=1


                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                        self.env.full_graph_dgl.norms[link_name][-3] += 1     
                        tp.append(len(self.env.full_graph_dgl.norms[link_name])-3)   


                    # CREATE LINKS  
                    # vehicle flow
                    src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                    # reverse flow
                    src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])

                    if phase[link_idx] == 'G':
                        self.env.full_graph_dgl.norms[link_name][2] +=1
                        tp.append(2)
                        self.env.full_graph_dgl.norms[phase_name][3] +=1
                        tp.append(3)                 

                    elif phase[link_idx] == 'g':
                        self.env.full_graph_dgl.norms[link_name][4] +=1                    
                        tp.append(4)    
                        self.env.full_graph_dgl.norms[phase_name][5] +=1                    
                        tp.append(5)                                  

                    elif phase[link_idx] == 'r':
                        self.env.full_graph_dgl.norms[link_name][6] +=1                    
                        tp.append(6)
                        self.env.full_graph_dgl.norms[phase_name][7] +=1                    
                        tp.append(7)    

                    elif phase[link_idx] == 'y':
                        self.env.full_graph_dgl.norms[link_name][8] +=1                    
                        tp.append(8)
                        self.env.full_graph_dgl.norms[phase_name][9] +=1                    
                        tp.append(9)    



                    for dir_idx in range(2):                        
                        lane_id = link[0][dir_idx]

                        if lane_id[:-2] not in self.env.full_graph_dgl.adresses_in_graph: 
                            edge_name = lane_id[:-2]
                            self.env.full_graph_dgl.adresses_in_graph[edge_name] = counter
                            self.env.full_graph_dgl.norms[edge_name] = [0]*21
                            self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = edge_name
                            node_type.append(3)
                            counter+=1


                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                            self.env.full_graph_dgl.norms[edge_name][-5] += 1     
                            tp.append(len(self.env.full_graph_dgl.norms[edge_name])-5)  




                        if lane_id not in self.env.full_graph_dgl.adresses_in_graph: 
                            # CREATE NODES 
                            self.env.full_graph_dgl.adresses_in_graph[lane_id] = counter
                            self.env.full_graph_dgl.norms[lane_id] = [0]*21
                            self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                            node_type.append(2)
                            counter+=1


                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                            self.env.full_graph_dgl.norms[lane_id][-4] += 1     
                            tp.append(len(self.env.full_graph_dgl.norms[lane_id])-4)  


                                # CREATE LINKS
                            src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                            # reverse flow
                            src.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])

                            self.env.full_graph_dgl.norms[edge_name][14] +=1
                            tp.append(14)
                            self.env.full_graph_dgl.norms[lane_id][15] +=1                        
                            tp.append(15)     



                        # CREATE LINKS 
                        src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                        # reverse flow
                        src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])


                        if dir_idx == 0:

                            self.env.full_graph_dgl.norms[lane_id][10] +=1
                            tp.append(10)
                            self.env.full_graph_dgl.norms[link_name][11] +=1                        
                            tp.append(11)     


                        elif dir_idx == 1:

                            self.env.full_graph_dgl.norms[lane_id][12] +=1
                            tp.append(12)
                            self.env.full_graph_dgl.norms[link_name][13] +=1                        
                            tp.append(13)     



        for destination, t in zip(dst,tp):
            norm.append([(1/self.env.full_graph_dgl.norms[self.env.full_graph_dgl.adresses_in_sumo[str(destination)]][t])])



        num_nodes = counter


        self.env.full_graph_dgl.add_nodes(num_nodes)
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        edge_type = torch.LongTensor(tp)
        edge_norm = torch.FloatTensor(norm).squeeze()
        node_type = torch.LongTensor(node_type).squeeze()

        self.env.full_graph_dgl.add_edges(src,dst)                        
        self.env.full_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                            
        self.env.full_graph_dgl.ndata.update({'node_type' : node_type})                            





    self.env.original_graphs = {}


    if 'tl_graph' in self.env.env_params.additional_params["generated_graphs"]:
        self.env.original_graphs["tl_graph"] = self.env.tl_graph_dgl 

    if 'lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
        self.env.original_graphs["lane_graph"] = self.env.lane_graph_dgl     

    if 'tl_lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
        self.env.original_graphs["tl_lane_graph"] = self.env.tl_lane_graph_dgl  

    if 'tl_connection_lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
        self.env.original_graphs["tl_connection_lane_graph"] = self.env.tl_connection_lane_graph_dgl  

    if 'full_graph' in self.env.env_params.additional_params["generated_graphs"]:
        self.env.original_graphs["full_graph"] = self.env.full_graph_dgl  


    for graph_name, graph in self.env.original_graphs.items():
        graph.nodes_lists = collections.OrderedDict()
        for node_type, identifier in graph.nodes_types.items():
            new_filt = partial(filt, identifier = identifier)
            graph.nodes_lists[node_type] = graph.filter_nodes(new_filt)


    if self.env.env_params.additional_params["veh_as_nodes"]:
        for graph_name, graph in self.env.original_graphs.items():
            if "lane" in graph_name or "full" in graph_name:
                for norm in graph.norms.values():
                    norm.extend([0,0,0])  

    for graph_name, graph in self.env.original_graphs.items():
        if graph_name != "lane_graph":
            graph.tl_subgraph = graph.subgraph(list(graph.nodes_lists['tl']))             







def setup_paths(self):
    for trip_name, paths in self.env.shortest_paths.items():
        counter = 0
        for route in paths:
            self.env.traci_connection.route.add(str(trip_name + "_" + str(counter)), route)  
            counter+=1

    self.env.entering_edges_probs = np.ones(len(self.env.entering_edges))
    self.env.leaving_edges_probs = np.ones(len(self.env.leaving_edges))
