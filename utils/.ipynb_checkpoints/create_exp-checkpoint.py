
# CREATE GLOBAL GRAPH (USE CODE IN "FROM HERE")

Lanes = {}
Agents = {}
    
    
    
    
#CREATE LANES INSTANCES
for lane_id in traci.lane.getIDList():
    
    
    
    inb_adj_vector = get_inb_adj_vector(graph,lane_id)
    outb_adj_vector = get_outb_adj_vector(graph,lane_id)
    distance_vector, discount_vector = get_distance_vector_lane(graph,lane_id)
    lanes[lane] = Lane(lane_id, inb_adj_vector, outb_adj_vector, discount_vector)

#CREATE AGENTS/TLs INSTANCES
for tl_id in traci.trafficlights.getIDList():
    distance_vector, discount_vector = get_discount_vector_tl(graph, tl_id)
    inbount_lanes, outbound_lanes, connections = get_controlled_lanes(graph, tl_id)
    Agents[tl_id] = Agent(tl_id, discount_vector, inbound_lanes, outbound_lanes, connections)




