from tensorboardX import SummaryWriter
import numpy as np
from utils.new_functions_exam import *

def init(env):
    
    
    env.graph_of_interest = env.env_params.additional_params['graph_of_interest']
    
    env.steps_done = 0 

    env.env_params.additional_params['tb_filename'] = str(env.env_params.additional_params['mode'] + "_" + env.env_params.additional_params['tb_filename'])

    if env.env_params.additional_params["random_objectives"]:
        env.objectives = ['column', 'line', 'full']

        for tl_id in env.Agents:  
            objective = np.random.choice(env.objectives)
            env.Agents[tl_id].objective = objective 



    env.model_init = False 

    #INITIALIZE MODEL 


    # configurations
    n_hidden = env.env_params.additional_params["n_hidden_layers"] # number of hidden units
    n_hidden_message = env.env_params.additional_params["n_hidden_message"]
    n_hidden_aggregation = env.env_params.additional_params["n_hidden_aggregation"]
    n_hidden_prediction = env.env_params.additional_params["n_hidden_prediction"]
    n_bases = -1 # use number of relations as number of bases
    n_hidden_layers = env.env_params.additional_params["n_hidden_layers"] # use 1 input layer, 1 output layer, no hidden layer
    n_epochs = env.env_params.additional_params["model_train_epochs"] # epochs to train
    lr = env.env_params.additional_params["learning_rate"] # learning rate
    l2norm = env.env_params.additional_params["l2_norm"] # L2 norm coefficient                
    num_classes = env.env_params.additional_params["n_classes"]      
    num_rels = len(list(env.original_graphs[env.graph_of_interest].norms.values())[0])
    rel_num_bases = env.env_params.additional_params["rel_num_bases"]


    num_nodes_network = len(env.original_graphs[env.graph_of_interest])
    env.num_nodes_network = num_nodes_network
    num_nodes = num_nodes_network * env.env_params.additional_params["batch_size"]
    num_tl_nodes_network = len(env.tl_subgraph_parent_nid)
    env.num_tl_nodes_network = num_tl_nodes_network
    num_tl_nodes = num_tl_nodes_network * env.env_params.additional_params["batch_size"]

    num_nodes_types = len(np.unique(env.original_graphs[env.graph_of_interest].ndata['node_type'].numpy()))
    if env.env_params.additional_params["veh_as_nodes"] and ("lane" in env.graph_of_interest or "full" in env.graph_of_interest):
        num_nodes_types +=1


    std_attention = env.env_params.additional_params["normalize attention"]




    nodes_types_num_bases = env.env_params.additional_params["node_types_num_bases"]
    hidden_layers_size = env.env_params.additional_params["nn_layers_size"]
    n_convolutional_layers = env.env_params.additional_params["n_convolutional_layers"]




    train_idx = np.array(range(num_tl_nodes))
    env.train_idx = train_idx
    train_limit = len(train_idx) // 5
    env.train_limit = train_limit
    val_idx = train_idx[:train_limit]   
    env.val_idx = val_idx
    train_idx = train_idx[train_limit:]
    env.train_idx = train_idx

    node_embedding_size = env.env_params.additional_params["nn_layers_size"]        



    state_vars = env.env_params.additional_params["state_vars"] + ["hid"]        

    """
    node_state_size = 0 

    for var in state_vars[:-1]:
        node_state_size += len(env.current_graphs[graph_of_interest].ndata[var][0])
    """

    node_state_size = env.node_state_size + env.env_params.additional_params["nn_layers_size"]

    separate_actor_critic = env.env_params.additional_params['separate_actor_critic']
    n_actions = env.env_params.additional_params["n_actions"]

    #print("node_state_size", node_state_size)

    prediction_size = env.env_params.additional_params["prediction_size"]    
    norm = env.env_params.additional_params["norm"]
    use_aggregation_module = env.env_params.additional_params["use_aggregation_module"]    
    use_message_module = env.env_params.additional_params["use_message_module"]    
    dropout = env.env_params.additional_params["dropout"]   
    num_propagations = env.env_params.additional_params["num_propagations"]
    resnet = env.env_params.additional_params["resnet"]
    """        
    env.model = Relational_Message_Passing_Framework(env.original_graphs[graph_of_interest], n_hidden = n_hidden_layers, hidden_layers_size = env.env_params.additional_params["nn_layers_size"], norm = env.env_params.additional_params["norm"], num_bases = -1, in_feat = env.env_params.additional_params['nn_layers_size']+2*8 +2,
                                     out_feat = env.env_params.additional_params['nn_layers_size'],
                                     is_bias = True,
                                     activation = F.relu)



# FRAMEWORK ELABORE

    env.model = Relational_Message_Passing_Framework2(num_nodes_types = num_nodes_types, nodes_types_num_bases = nodes_types_num_bases, node_state_dim = node_state_size + node_embedding_size, node_embedding_size = node_embedding_size, num_rels = num_rels, n_hidden = n_hidden, hidden_layers_size = hidden_layers_size, prediction_size = prediction_size, num_propagations = 1, rel_num_bases = -1, norm = True, is_bias = True, activation = F.relu, use_aggregation_module = False, is_final_convolutional_layer = True)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    

    rl_learner_type = "critic"
    
    
    
    
    
    

# POSSIBILITE DE CREER UNE CLASSE QUI INCLUE PLUSIEURS FRAMEWORK AVEC DES PARAMETRES DIFFERENTS POUR FAIRE UNE CONVOLUTION
    env.benchmark = Convolutional_Message_Passing_Framework(use_attention = False, separate_actor_critic = separate_actor_critic, n_actions = n_actions, rl_learner_type = rl_learner_type, std_attention = std_attention, state_vars = state_vars, n_convolutional_layers = n_convolutional_layers, num_nodes_types = num_nodes_types, nodes_types_num_bases = nodes_types_num_bases, node_state_dim = node_state_size, node_embedding_dim = node_embedding_size, num_rels = num_rels, n_hidden_message = n_hidden_message, n_hidden_aggregation = n_hidden_aggregation, n_hidden_prediction = n_hidden_prediction, hidden_layers_size = hidden_layers_size, prediction_size = prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = True, activation = F.elu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, dropout = dropout, resnet = resnet)



    env.model = Convolutional_Message_Passing_Framework(use_attention = True, separate_actor_critic = separate_actor_critic, n_actions = n_actions, rl_learner_type = rl_learner_type, std_attention = std_attention, state_vars = state_vars, n_convolutional_layers = n_convolutional_layers, num_nodes_types = num_nodes_types, nodes_types_num_bases = nodes_types_num_bases, node_state_dim = node_state_size, node_embedding_dim = node_embedding_size, num_rels = num_rels, n_hidden_message = n_hidden_message, n_hidden_aggregation = n_hidden_aggregation, n_hidden_prediction = n_hidden_prediction, hidden_layers_size = hidden_layers_size, prediction_size = prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, is_bias = True, activation = F.elu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, dropout = dropout, resnet = resnet)






    if env.env_params.additional_params['mode'] == 'test':
        env.model.load_state_dict(torch.load(env.env_params.additional_params['trained_model_path'] + "/" + env.env_params.additional_params['tb_foldername'] + ".pt"))
        env.model.eval()
        

    elif env.env_params.additional_params['mode'] == 'train':
        env.model.train()
        env.benchmark.train()





    """
    if torch.cuda.device_count() > 1:
        env.model = nn.DataParallel(env.model)
    """

    # optimizer
    env.model.optimizer = torch.optim.Adam(env.model.parameters(), lr=lr, weight_decay=l2norm)                 
    env.model.original_graphs  = env.original_graphs
    env.model.env_params = env.env_params
    env.model.tl_subgraph_parent_nid = env.tl_subgraph_parent_nid
    env.model.Agents = env.Agents
    env.model.num_nodes_network = env.num_nodes_network
    env.model.num_tl_nodes_network = env.num_tl_nodes_network
    env.model.n_workers = env.n_workers
    
    env.benchmark.optimizer = torch.optim.Adam(env.benchmark.parameters(), lr=lr, weight_decay=l2norm)                 
    env.benchmark.original_graphs  = env.original_graphs
    env.benchmark.env_params = env.env_params
    env.benchmark.tl_subgraph_parent_nid = env.tl_subgraph_parent_nid
    env.benchmark.Agents = env.Agents
    env.benchmark.num_nodes_network = env.num_nodes_network
    env.benchmark.num_tl_nodes_network = env.num_tl_nodes_network
    env.benchmark.n_workers = env.n_workers    

    
    return env