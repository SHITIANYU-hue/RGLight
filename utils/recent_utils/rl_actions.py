def rl_actions():
    
    activated_tls = []
    self.env.Actions = collections.OrderedDict()
    self.env.Targets = collections.OrderedDict()

    labels = [0]*self.env.original_graphs["tl_graph"].number_of_nodes() 
    actions = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
    choices = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
    forced_0 = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
    forced_1 = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()

    for tl_id in self.env.Agents:
        self.env.Targets[tl_id] = self.env.Agents[tl_id].get_reward(self.env.reward_vector)
        if self.env.env_params.additional_params["std_nb_veh"]:
            n_veh = 0
            for lane_id in self.env.Agents[tl_id].complete_controlled_lanes:
                if 'c' not in lane_id:
                    n_veh += self.env.Lanes[lane_id].state[0]
            self.env.Targets[tl_id] /= max(n_veh,1)

        labels[list(self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = self.env.Targets[tl_id]


    if self.env.seed == 0 :
        print("step", self.env.step_counter)


    # Q_LEARNING
    if self.env.env_params.additional_params['Policy_Type'] == "Q_Learning" :

        self.env.request_end.send((self.env.current_graphs[self.env.graph_of_interest],self.env.original_graphs["tl_graph"].number_of_nodes())  if not self.env.baseline else 'N/A')

        if not self.env.baseline:
            actions = self.env.request_end.recv()
            actions = list(actions)

            if not self.env.greedy and not self.env.env_params.additional_params['gaussian_mixture']:
                if self.env.eps_threshold > self.env.env_params.additional_params['EPS_END']:
                    self.env.eps_threshold = self.env.eps_threshold * self.env.env_params.additional_params['EPS_DECAY']
                if self.env.step_counter % 50 == 0 and self.env.seed == 0:
                    print("eps_threshold :", self.env.eps_threshold)

                for tl_id in self.env.Agents:
                    sample = random.random()
                    if sample < self.env.eps_threshold:
                        actions[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] =  torch.randint(low=0, high=2, size=(1,1)).squeeze() 

        else:
            _ = self.env.request_end.recv()


    # ACTOR CRITIC
    elif 'critic' in self.env.env_params.additional_params['Policy_Type'].lower() and 'actor' in  self.env.env_params.additional_params['Policy_Type'].lower():                
        self.env.request_end.send((self.env.current_graphs[self.env.graph_of_interest], labels))

        actions = self.env.request_end.recv() 
        actions = list(actions.numpy())


    # ONLY CRITIC
    elif 'critic' in self.env.env_params.additional_params['Policy_Type'].lower():                
        for tl_id in self.env.Agents:
            self.env.Actions[tl_id] = 0
            if self.env.Agents[tl_id].time_since_last_action >= 5:
                self.env.Actions[tl_id] = 1 
            self.env.Targets[tl_id] = self.env.Agents[tl_id].get_reward(self.env.reward_vector)      




    # CONSTRAINTS 
    if not self.env.tested == 'classic':
        for tl_id in self.env.Agents:
            if self.env.Agents[tl_id].time_since_last_action+1 >= self.env.env_params.additional_params["min_time_between_actions"] and ((self.env.Agents[tl_id].time_since_last_action+1 - self.env.env_params.additional_params["min_time_between_actions"]) % self.env.env_params.additional_params["time_between_actions"] == 0 or self.env.Agents[tl_id].time_since_last_action+1 == self.env.env_params.additional_params["min_time_between_actions"]):


                if 'y' in self.env.Agents[tl_id].current_phase.lower():
                    if self.env.Agents[tl_id].time_since_last_action +1 < self.env.env_params.additional_params["yellow_duration"]:
                        self.env.Actions[tl_id] = 0
                        forced_0[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = 1                

                    else:
                        self.env.Actions[tl_id] = 1
                        forced_1[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = 1                

                else: # WE GET "CHOSEN" ACTIONS FROM THE ACTION VECTOR 
                    if self.env.tested == 'strong_baseline':
                        if self.env.Agents[tl_id].nb_stop_inb > self.env.Agents[tl_id].nb_mov_inb :
                            self.env.Actions[tl_id] = 1
                        else:
                            self.env.Actions[tl_id] = 0
                    elif self.env.baseline: 
                        self.env.Actions[tl_id] = 1
                    else:
                        self.env.Actions[tl_id] = actions[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])].item() 
                    choices[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = 1



            else:
                self.env.Actions[tl_id] = 0
                forced_0[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = 1                


            if self.env.env_params.additional_params['correct_actions'] and self.env.env_params.additional_params['Policy_Type'] == "Q_Learning":
                # IF USING "REAL ACTIONS ONLY" WE TRAIN USING THE ACTIONS VECTOR WITH ACTIONS THAT ARE ACTUALLY TAKEN                          

                actions[list(self.env.self.env.current_graphs[self.env.graph_of_interest].tl_subgraph.parent_nid).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = self.env.Actions[tl_id]


            self.actions_counts[self.env.Actions[tl_id]] +=1

        # SAVE STEP 
        if self.env.step_counter > self.env.env_params.additional_params['wait_n_steps']:            
            if 'critic' in self.env.env_params.additional_params['Policy_Type'].lower() and 'actor' in self.env.env_params.additional_params['Policy_Type'].lower():
                pass                     
            elif not self.env.baseline:
                self.env.all_graphs.append([self.env.current_graphs[self.env.graph_of_interest],labels, actions, choices, forced_0, forced_1]) # choices      

    elif self.env.tested == 'classic':
        self.env.Actions = None

    return self.env.Actions        







