import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from statistics import mean 

def results(n_workers, params, baseline_reward_queue = None, greedy_reward_queue = None,  tested_learner_ends = None, tested = None, reward_queues = None):


    # IN THE CASE OF AN ACTOR CRITIC, THE COMPUT DOES THE LEARNING 
    global_rewards = []
    global_baseline_rewards = []
    gr_counter = 0
    gbr_counter = 0 
    train_counter = 0 
    max_reward = None
    writer = SummaryWriter(params["tb_foldername"])
    writer.add_custom_scalars_multilinechart(['general_train_loss', 'general_test_loss'], title = 'General Train/Test Loss')
    # MULTILINE CHARTS FOR TENSORBOARD
    if params['mode'] == 'train':
        writer.add_custom_scalars_multilinechart(['general_train_loss', 'general_test_loss'], title = 'General Train/Test Loss')
    elif params['mode'] == 'test':
        writer.add_custom_scalars_multilinechart([str(x) for x in tested], title = 'Rewards Comparison')   

    for k,v in params.items():
        if type(v) == list:
            for idx, elem in enumerate(v):
                if idx == 0:
                    value = str(elem)
                else:
                    value += str(" ||  " + str(elem))

            writer.add_text(k, value, global_step=None, walltime=None)
        else:
            writer.add_text(k, str(v), global_step=None, walltime=None)


    # INITIALIZE REWARDS HOLDERS
    global_rewards = []
    if params['save_extended_training_stats']:
        global_queues = []
        global_delays = []
        global_co2s = []
    global_baseline_rewards = []
    global_greedy_rewards = []

    # REMOVE BECAUSE 0 == GREEDY
    if greedy_reward_queue is not None:
        try:
            memory_queues.pop(0)
            reward_queues.pop(0)
        except Exception as e:
            print(e)
    # REMOVE BECAUSE 1 == BASELINE
    if baseline_reward_queue is not None:
        try:
            memory_queues.pop(1)
            reward_queues.pop(1)
        except Exception as e:
            print(e)        

    def collate(samples):
        return torch.stack((samples), 2).squeeze()
    #print("2")        
    while True:
        reward_to_save = None
        log_probs_list = []
        values_list    = []
        states_list    = []
        actions_list   = []
        rewards_list  = []
        entropy = 0   
        counter = 0


        load_counter = 0
        if params['mode'] == 'test':
            for idx, tested_end in tested_ends.items():
                while tested_end.poll():
                    reward, queues, co2 = tested_end.recv()
                    tested_rewards[tested[idx]].append(float(reward))
                    tested_queues[tested[idx]].append(float(queues))
                    tested_co2[tested[idx]].append(float(co2))

                if len(tested_rewards[tested[idx]]) >= params['n_avg_test']:
                    writer.add_scalar("delay "+ str(idx),mean(tested_rewards[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    writer.add_scalar("queue length "+ str(idx),mean(tested_queues[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    writer.add_scalar("CO2 emissions "+ str(idx),mean(tested_co2[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    """
                    writer.add_scalar("delay "+ str(tested[idx]),mean(tested_rewards[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    writer.add_scalar("queue length "+str(tested[idx]),mean(tested_queues[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    writer.add_scalar("CO2 emissions "+str(tested[idx]),mean(tested_co2[tested[idx]][-params['n_avg_test']:])/float(params['num_steps_per_experience']),tested_counters[tested[idx]]) 
                    """
                    tested_rewards[tested[idx]] = []
                    tested_queues[tested[idx]] = []
                    tested_co2[tested[idx]] = []
                    tested_counters[tested[idx]] += 1




        #print("3")                        
        if params['mode'] == 'train':                        
            # RECEIVE REWARDS                    
            if greedy_reward_queue is not None:
                while not greedy_reward_queue.empty():
                    global_greedy_reward = greedy_reward_queue.get()
                    global_greedy_rewards.append(global_greedy_reward)           

            #print("a")

            for idx, reward_queue in reward_queues.items(): 
                if reward_queue.poll():
                    #print("b")
                    if params['save_extended_training_stats']:
                        global_reward, global_delay, global_queue, global_co2 = reward_queue.recv()
                        global_rewards.append(global_reward)
                        global_queues.append(global_queue)
                        global_delays.append(global_delay)
                        global_co2s.append(global_co2)
                    else:
                        global_reward = reward_queue.recv()
                        global_rewards.append(global_reward)


                # WRITE METRICS
                #print("yo")
                if len(global_rewards) >= n_workers:
                    #print("YOLO")
                    if params['save_extended_training_stats']:
                        reward_to_save = mean(global_delays[-n_workers:])/(params['num_steps_per_experience'])
                        writer.add_scalar('global delay (average of ' + str(n_workers) + ' experiments)', reward_to_save, gr_counter)
                        global_delays = []
                        reward_to_save = mean(global_queues[-n_workers:])/(params['num_steps_per_experience'])
                        writer.add_scalar('global queue (average of ' + str(n_workers) + ' experiments)', reward_to_save, gr_counter)
                        global_queues = []
                        reward_to_save = mean(global_co2s[-n_workers:])/(params['num_steps_per_experience'])
                        writer.add_scalar('global co2 (average of ' + str(n_workers) + ' experiments)', reward_to_save, gr_counter)
                        global_co2s = []
                    reward_to_save = mean(global_rewards[-n_workers:])/(params['num_steps_per_experience'])
                    writer.add_scalar('global reward (average of ' + str(n_workers) + ' experiments)', reward_to_save, gr_counter)
                    global_rewards = []
                    gr_counter +=1


                    if greedy_reward_queue is not None:
                        writer.add_scalar('global greedy reward', mean(global_greedy_rewards)/(params['num_steps_per_experience']), ggr_counter)
                        global_greedy_rewards = []
                        ggr_counter +=1

