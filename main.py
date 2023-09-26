import os
import time
import datetime
import pickle
import yaml
import threading
import logging
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.clientsFile.clientManager import ClientManager
from src.utils import launch_tensor_board, graph_Q, graph_combine, graph_budget, graph_many_Q, graph_many_budget, graph_many_utility, graph_util, calculate_l, graph_combine2, graph_results

if __name__ == "__main__":
    # read configuration file
    with open('./config.yaml') as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    rep_config = configs[7]["reputation_config"]
    bad_client_config = configs[8]["badClient_config"]
    client_cost_per_data = rep_config['client_cost_per_data']
    if global_config['USE_MPS'] and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("using mps")
    elif global_config['USE_CUDA'] and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                device = torch.device('cuda:'+str(i))
                torch.cuda.set_device(i)
                logging.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
                break
            except Exception as e:
                assert i != torch.cuda.device_count()-1, 'Can not find a feasible GPU'
        print("using cuda")
    else:
        device = torch.device('cpu')
        print("using cpu")
   
    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"], str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
        ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt=" %Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)
     
    for config in configs: 
        print(config); logging.info(config)
    print()
     
    '''Initialize Federated Learning Servers
       Client Manager will init the clients'''
    clientManager = ClientManager(writer,device, data_config,fed_config,optim_config, client_cost_per_data) 
    clientManager.setup()
    # clientManager.set_client_type("Market")
    clients = clientManager.get_clients()
    test_data, eval_dataloader = clientManager.get_evaluate_dataloader()
    central_server_1 = Server(1, writer,device, model_config, global_config, init_config, fed_config, optim_config, rep_config, bad_client_config, clients, data_config, test_data, eval_dataloader)

    # clientManager_2 = ClientManager(writer,device, data_config,fed_config,optim_config, client_cost_per_data)
    # clientManager_2.setup()
    # clientManager_2.set_client_type("Randomised")
    # clients = clientManager_2.get_clients()
    # central_server_2 = Server(2, writer,device, model_config, global_config, init_config, fed_config, optim_config, rep_config, bad_client_config, clients, data_config, test_data, eval_dataloader, V=5)
    # central_server_2.set_incentive_type('FIFL')
    
    # clientManager_3 = ClientManager(writer,device, data_config,fed_config,optim_config, client_cost_per_data)
    # clientManager_3.setup()
    # clientManager_3.set_client_type("Baseline")
    # clients = clientManager_3.get_clients()
    # central_server_3 = Server(3, writer,device, model_config, global_config, init_config, fed_config, optim_config, rep_config, bad_client_config, clients, data_config, test_data, eval_dataloader, V=8)
    # central_server_2.set_incentive_type('SV')
    
    central_server_1.setup()
    # central_server_2.setup()
    # central_server_3.setup()
    l_dict = {}
    l_1=[] 
    l_2=[]
    l_3=[]
    num_rounds = fed_config["R"]
    '''Start federated learning'''
    pool = Pool(processes=cpu_count())         
    for r in range(num_rounds):    
        
        client_bids = clientManager.get_client_bids()
        client_visit_arr = clientManager.randomize_visits()
    
        #if non randmoized visists
        # client_visit_arr = clientManager.get_visit_arr()
        
        if r>=1:
            l_dict['l_1'] = central_server_1.get_q_square()
            # l_dict['l_2'] = central_server_2.get_q_square()
            # l_dict['l_3'] = central_server_3.get_q_square()

            l_1.append(calculate_l(l_dict, 'l_1'))
            # l_2.append(calculate_l(l_dict, 'l_2'))
            # l_3.append(calculate_l(l_dict, 'l_3'))
            # central_server_1.set_overall_l2norm(l_1[-1])
            # central_server_2.set_overall_l2norm(l_2[-1])
            # central_server_3.set_overall_l2norm(l_3[-1])
        # print(client_bids)
        # pool.apply_async(central_server_1.fit(r, client_visit_arr, clients_bids=client_bids)) 
        central_server_1.fit(r, client_visit_arr, clients_bids=client_bids) 
        # pool.apply_async(central_server_2.fit(r, client_visit_arr[1])) 
        # pool.apply_async(central_server_3.fit(r, client_visit_arr[2])) 

    # for r in range(num_rounds):    
    #     client_visit_arr_2 = clientManager_2.randomize_visits()
    #     central_server_2.fit(r, client_visit_arr_2)

    # for r in range(num_rounds):    
    #     client_visit_arr_3 = clientManager_3.randomize_visits()
    #     central_server_3.fit(r, client_visit_arr_3)

    # save resulting losses and metrics
    with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
        pickle.dump(central_server_1.results, f)


    #save the final model
    central_server_1.save_model()
    # central_server_2.save_model()
    # central_server_3.save_model()

    """ Print out """
    print("--Budget1--")
    print(central_server_1.get_PR_budget_hist())
    print("--Utils1--")
    print(central_server_1.get_utility_history())
    print("Q_1 is")
    print(central_server_1.get_Q_hist())
    print("Acc is")
    print(central_server_1.get_results()["accuracy"])
    
    # print("--Budget2--")
    # print(central_server_2.get_PR_budget_hist())
    # print("--Utils2--")
    # print(central_server_2.get_utility_history())

    # print("--Budget3--")
    # print(central_server_3.get_PR_budget_hist())
    # print("--Utils3--")
    # print(central_server_3.get_utility_history())
   
    # print("L_1 history is")
    # print(l_1)
    # print("L_2 history is")
    # print(l_2)
    # print("L_3 history is")
    # print(l_3)


    # graph_many_Q(central_server_1.get_Q_hist(), central_server_2.get_Q_hist(), central_server_3.get_Q_hist())
    # graph_many_utility(central_server_1.get_utility_history(), central_server_2.get_utility_history(), central_server_3.get_utility_history())
    # graph_many_budget(central_server_1.get_PR_budget_hist(), central_server_2.get_PR_budget_hist(), central_server_3.get_PR_budget_hist())
    
    graph_combine(1, central_server_1.get_PR_budget_hist(),central_server_1.get_Q_hist(), central_server_1.get_utility_history())
   #results is pure model performance
    graph_results(central_server_1.get_results())
    # graph_combine(2, central_server_2.get_PR_budget_hist(),central_server_2.get_Q_hist(), central_server_2.get_utility_history())
    # graph_combine(3, central_server_3.get_PR_budget_hist(),central_server_3.get_Q_hist(), central_server_3.get_utility_history())
    
    # graph_combine2(1, central_server_1.get_PR_budget_hist(),l_1, central_server_1.get_utility_history())
    # graph_combine2(2, central_server_2.get_PR_budget_hist(),l_2, central_server_2.get_utility_history())
    # graph_combine2(3, central_server_3.get_PR_budget_hist(),l_3, central_server_3.get_utility_history())

    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit()
