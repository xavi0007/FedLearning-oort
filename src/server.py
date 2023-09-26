from audioop import avg
import copy
import gc
import logging
from pickletools import optimize
import random
# from this import d

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict
from .svfl.svfl import calculate_sv
from .fifl.fifl import calculate_reward
from .rrafl.rrafl import *
from .models import *
from .utils import *
from .oort.oort import create_training_selector, create_testing_selector
from .oort.clientSampler_oort import clientSampler_Oort

logger = logging.getLogger(__name__)


class Server(object):
    """Each server class will instantiate required amount of clients needed to complete FL task.
    1. init Server with config defined in config.yaml
    2. Server to init weights to the choosen model and distribute it to all clients, Server selects some clients to contribute the model weight updates based on reputation
    3. Communication round starts
    4. Client to contribute their data (i.i.d or non-i.i.d)
    5. (optional) Clients maybe malicious or simply honest or struggling
    6. Server aggrgregates the weights based on FedAvg method
    
    Key Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda", "mps").
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, id, writer, device, model_config={}, global_config={}, init_config={}, fed_config={}, optim_config={}, rep_config={}, bad_client_config={}, _clients=[], data_config={}, test_data=[], data_loader=None, V=1):
        self.id = id
        self.clients = _clients        
        self.dataloader = data_loader
        self.data = test_data
        self._round = 0
        self.writer = writer
        self.device = device
        
        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.iid = data_config["iid"]       
        
        self.model = eval(model_config["name"])(**model_config)
        self.seed = global_config["seed"]
        self.global_grad = None
    
        self.num_clients = fed_config["K"] #number of clients, irrelevant, set to 0
        self.fraction = fed_config["C"]  #Fraction of sampled clients to be used for aggregation, set to 1
        self.num_rounds = fed_config["R"] #communication rounds
        self.local_epochs = fed_config["E"] #number of epochs in device
        self.batch_size = fed_config["B"] #batch size 
        self._V = fed_config["V"] #weight param for lyapunov
        self.multiplier_g = fed_config["multiplier_g"]
        self.criterion = fed_config["criterion"]
        self.incentive_type = fed_config["incentive"]
        self.total_budget = fed_config["total_budget"]
        self.exe_cost_per_client = fed_config["exe_cost_per_client"]
        self.time_required = fed_config["time_required"]
        self.init_config = init_config
        
        self.rep_threshold = rep_config['reputation_threshold']
        self.client_cost_per_data = rep_config['client_cost_per_data']
        
        self._Q = 0
        self.l2norm = 0
        self.hired_px_hist = []
        self.avg_price = 1
        self.opt_bid = 0
        self.total_bid = 0
    
        #dictionary; idx = client_idx,  value = [rep, alpha, beta, most recent accuracy]
        self.client_rep_list = {}
        #for oort
        self.sampled_cliet_set = set()
        self.PR_budget = 0
        self.hire_budget = 0
        self.used_PR_budget = 0
        self.invoke_cost = 0

        # self.hire_count_prev = 0
        self.hire_count_now = 0
        
        # self.num_bad_clients = bad_client_config['bad_K'] # number of bad clients included in count for num_clients
        # self.client_attack = bad_client_config['attack']
        # self.client_attack_prop = bad_client_config['prop_attack']

        self.results = {"loss": [], "accuracy": []} 
        self.utility_history = []
        self.PR_used_budget_hist = []
        self._Q_hist = []


        self.stale_threshold = 0
        self.staleness = 0
        self.learner_staleness = {l: 0 for l in range(self.num_clients)}
        self.learner_local_step = {l: 0 for l in range(self.num_clients)}
        self.learner_cache_step = {l: 0 for l in range(self.num_clients)}
        self.clock_factor = 1

        #clients on hold out 
        self.pendingWorkers = {}
        self.exploredPendingWorkers = []
        self.virtualClientClock = {}
        self.avgUtilLastEpoch = 0.
        self.clientsLastEpoch = []
        self.sumDeltaWeights = []
        self.last_sampled_clients = None

        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.clock_factor = 1
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0
        
        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, self.device, **self.init_config, )

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()
        self.transmit_model()
        # if self.incentive_type == "Reverse" or self.incentive_type == "ReverseMod":
        self.basic_PR_budget = self.total_budget / self.num_rounds
        self.PR_budget = self.basic_PR_budget

        if self.incentive_type == 'oort':
            self.oort_train_selector = create_training_selector()
            self.oort_test_selector = create_testing_selector()
            #simple init to feedback
            feedbacks = {'reward': 0, 'duration':0 , 'time_stamp':1, 'count': 0, 'status': True}
            #regist all clients first into the arms
            for client in self.clients:
                self.oort_train_selector.register_client(client.get_id(), feedbacks)

    def set_incentive_type(self, type):
        self.incentive_type = type

    #-------------------CLIENT RELATED FUNCTIONS-----------------------------------------------------
    """Step 1.Select clients based on selected criteria""" 
    def sample_clients(self, client_visit_arr, clients_bids):      
        message = f"[Server: {str(self.id)}, Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        num_sampled_clients = max(int(self.fraction * (self.num_clients)), 1)
        sampled_client_indices = []
            
        #shuffle client rep list to visit clients randomly
        # if (len(self.client_rep_list) != 0):
        #     self.client_rep_list = {k:self.client_rep_list[k] for k in random.sample(list(self.client_rep_list.keys()), len(self.client_rep_list))}   
        # print(self.client_rep_list)

        # Sample clients based on their reputation from federation POV
        if (self._round >= 1):
            if self.incentive_type == "SV":
                total_asking_price = 0
                #build another list of clients that has higher rep than threshold
                for key in (client_visit_arr):
                    if(key in self.client_rep_list.keys()):
                        if (self.client_rep_list[key][0] >= self.rep_threshold):
                            sampled_client_indices.append(key)
                            self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            total_asking_price += self.clients[key].get_asking_price() 
                        else:
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                sv = calculate_sv(sampled_client_indices, self.client_rep_list)
                #incentive based on contribution value
                temp_budget= sum(sv.values())*0.5*total_asking_price
                #normalize it such that it can be compared to other methods, x-xmin / xmax-xmin, xmin = 0.8 xmax =3
                self.used_PR_budget = (temp_budget-0.8) / 2.2
            elif self.incentive_type == "Greedy":
                #greedy only hires the lower 25% price of clients
                count = 0
                asking_px_arr = []
                for key in (client_visit_arr):
                    asking_px_arr.append(self.clients[key].get_asking_price())
                cut_off_price = np.percentile(asking_px_arr, 15)
                for key in (client_visit_arr):
                    #Check budget balance and reputation
                    asking_px= self.clients[key].get_asking_price() 
                    # print(f'client {key} has price of {asking_px}') 24 is 40% of 60
                    if(key in self.client_rep_list.keys() and count < 9):
                        if (asking_px <= cut_off_price):
                            self.used_PR_budget += asking_px
                            #reply to clients
                            self.hired_px_hist.append(asking_px)
                            self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            count += 1
                            sampled_client_indices.append(key)
                        else: 
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
            elif self.incentive_type == "FIFL":
                #incentive based on update gradient
                for key in (client_visit_arr):
                    self.clients[key].set_algo_type("FIFL")
                    curr_grad = self.clients[key].get_grad() 
                    #Check budget balance and reputation                    
                    if(key in self.client_rep_list.keys()):
                        if (self.client_rep_list[key][0] >= self.rep_threshold):
                            incentive_px = calculate_reward(self.client_rep_list[key][0], 2, curr_grad, self.global_grad, client_visit_arr, self.clients)
                            self.used_PR_budget += incentive_px
                            #reply to clients
                            self.hired_px_hist.append(incentive_px)
                            self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        else: 
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
            elif self.incentive_type == "UCB":
                #incentive based on update gradient
                for key in (client_visit_arr):
                    self.clients[key].set_algo_type("UCB")                    
                    num_selected = self.clients[key].get_num_selected()
                    #calculate UCB 
                    #Check budget balance and UCB                    
                    if(key in self.client_rep_list.keys()):
                        if (self.client_rep_list[key][0] >= self.rep_threshold):
                            #calculate UCB 
                            #############################
                            incentive_px = self.clients[key].get_price()
                            #check if budget allow
                            if (self.used_PR_budget + incentive_px ) <= self.basic_PR_budget: 
                                #reply to clients
                                self.hired_px_hist.append(incentive_px)
                                self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        else: 
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
            elif self.incentive_type == "RRAFL":
                #get unit reputation bid 
                rep_bid_dict = {}
                for key in (client_visit_arr):
                    if(key in clients_bids.keys()):
                        self.clients[key].set_algo_type("RRAFL")
                        asking_px = self.clients[key].get_asking_price()
                        q_client = asking_px/self.client_rep_list[key][0]
                        rep_bid_dict[key] = q_client
                # sort by value
                sorted_rep_bid_list= sorted(rep_bid_dict.items(), key=lambda x:x[1])
                #convert to dict
                sorted_rep_bid_dict = dict(sorted_rep_bid_list)

                #find the selected group, all temp variables
                for key in (sorted_rep_bid_dict.keys()): 
                    if key+1 >= len(sorted_rep_bid_dict):
                        q_client_next = self.clients[key].get_asking_price() / self.client_rep_list[key][0]
                    else:
                        #use next unit price 
                        q_client_next = self.clients[key+1].get_asking_price() / self.client_rep_list[key][0]
                    incentive_px = self.client_rep_list[key][0] * q_client_next +  self.exe_cost_per_client
                    if(self.used_PR_budget + incentive_px  <= self.PR_budget):
                       self.used_PR_budget = self.used_PR_budget + incentive_px
                       #bookeeping
                       sampled_client_indices.append(key)
                       self.hired_px_hist.append(incentive_px)
                       self.clients[key].send_message(["Accept", self.cal_avg_price(), min(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), min(self.hired_px_hist)])
            elif self.incentive_type == "Uniform":
                for key in (client_visit_arr):
                    #Check budget balance and reputation
                    asking_px= self.clients[key].get_asking_price() 
                    # print(f'client {key} has price of {asking_px}')
                    if(key in self.client_rep_list.keys()):
                        if (self.client_rep_list[key][0] >= self.rep_threshold):
                            self.used_PR_budget += asking_px
                            #reply to clients
                            self.hired_px_hist.append(asking_px) 
                            self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        else: 
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
            elif self.incentive_type == "Vanilla": #vanilla reverse auction
                min_bid = min(clients_bids, key=clients_bids.get)
                for key in (client_visit_arr):
                    #first come first serve if many low bidders, don't want to starve if only pick high rep
                    if(key in clients_bids.keys()):
                        asking_px = self.clients[key].get_asking_price()
                        #if pariticpant has the lowest bid and budget allows
                        incentive_px = asking_px + self.exe_cost_per_client
                        if (asking_px == clients_bids[min_bid]) and ( (self.used_PR_budget + incentive_px ) <= self.basic_PR_budget) : 
                            self.hired_px_hist.append(asking_px) 
                            self.used_PR_budget += incentive_px
                            if len(self.hired_px_hist) == 0:
                                self.clients[key].send_message(["Accept", self.cal_avg_price(), 3])
                            else:
                                self.clients[key].send_message(["Accept", self.cal_avg_price(),  min(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        #reject participant
                        else:
                            #no previous participants
                            if len(self.hired_px_hist) == 0:
                                self.clients[key].send_message(["Decline", self.cal_avg_price(), 3])
                            else:
                                self.clients[key].send_message(["Decline", self.cal_avg_price(), min(self.hired_px_hist)])
            #normal reverse auction with no boosted income, select only the cheapest participants                
            elif self.incentive_type == "Reverse":
                min_bid = min(clients_bids, key=clients_bids.get)
                print("min bid is ", clients_bids[min_bid])
                for key in (client_visit_arr):
                    #first come first serve if many low bidders, don't want to starve if only pick high rep
                    if(key in clients_bids.keys()):
                        asking_px = self.clients[key].get_asking_price()
                        #if pariticpant has the lowest bid and budget allows
                        incentive_px = asking_px + self.exe_cost_per_client
                        if (asking_px == clients_bids[min_bid]) and ( (self.used_PR_budget + incentive_px ) <= self.basic_PR_budget) : 
                            self.hired_px_hist.append(asking_px) 
                            self.used_PR_budget += incentive_px
                            if len(self.hired_px_hist) == 0:
                                self.clients[key].send_message(["Accept", self.cal_avg_price(), 3])
                            else:
                                self.clients[key].send_message(["Accept", self.cal_avg_price(),  min(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        #reject participant
                        else:
                            #no previous participants
                            if len(self.hired_px_hist) == 0:
                                self.clients[key].send_message(["Decline", self.cal_avg_price(), 3])
                            else:
                                self.clients[key].send_message(["Decline", self.cal_avg_price(), min(self.hired_px_hist)])
            #reverse auction mechanism but federation have more expenditure budget
            elif self.incentive_type == "ReverseMod":
                #shortcut to optimize code
                if self.hire_budget <= 0:
                    sampled_client_indices = []
                    return sampled_client_indices
                for key in (client_visit_arr):
                    if(key in clients_bids.keys()):
                        asking_px = self.clients[key].get_asking_price() 
                        incentive_px = asking_px + self.exe_cost_per_client
                        if(key in self.client_rep_list.keys()):
                            if (self.hire_budget -incentive_px >= 0) and (self.client_rep_list[key][0] >= self.rep_threshold) : 
                                self.hired_px_hist.append(asking_px) 
                                self.hire_budget =  self.hire_budget - incentive_px
                                # for bookkepping of values
                                self.used_PR_budget = self.used_PR_budget + asking_px + self.exe_cost_per_client
                                if len(self.hired_px_hist) == 0:
                                    self.clients[key].send_message(["Accept", self.cal_avg_price(), 3])
                                else:
                                    self.clients[key].send_message(["Accept", self.cal_avg_price(),  min(self.hired_px_hist)])
                                sampled_client_indices.append(key)
                            else:
                                if len(self.hired_px_hist) == 0:
                                    self.clients[key].send_message(["Decline", self.cal_avg_price(), 3])
                                self.clients[key].send_message(["Decline", self.cal_avg_price(), min(self.hired_px_hist)])
            elif self.incentive_type == "oort":       
                #Select clients with pacer, the function takes care of the 2 exploitation method and the exploration by speed
                sampled_client_indices = self.oort_train_selector.select_participant(len(client_visit_arr))
                total_per_round_asking_price = 0
                for client_idx in client_visit_arr:
                    if client_idx in sampled_client_indices:
                        #the price dont actually matter here. just need to send the message, and keep track for comparison
                        self.clients[client_idx].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[client_idx].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    total_per_round_asking_price += self.clients[client_idx].get_asking_price() 
                    
                self.used_PR_budget += total_per_round_asking_price
            else:
                for key in (client_visit_arr):
                    #Check budget balance and reputation
                    asking_px= self.clients[key].get_asking_price() 
                    # print(f'client {key} has price of {asking_px}')
                    if(key in self.client_rep_list.keys()):
                        if (self.used_PR_budget + asking_px >= self.PR_budget) and (self.client_rep_list[key][0] >= self.rep_threshold):
                            self.used_PR_budget += asking_px
                            #reply to clients
                            self.hired_px_hist.append(asking_px)
                            self.clients[key].send_message(["Accept", self.cal_avg_price(), max(self.hired_px_hist)])
                            sampled_client_indices.append(key)
                        else: 
                            self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
                    else:
                        self.clients[key].send_message(["Decline", self.cal_avg_price(), max(self.hired_px_hist)])
        #cold start, not needed in new algorithm
        else:
            # sample clients randommly because don't know who are the clients i.e. cold start problem, note: these clients will be addeed during evaluation stage to rep list
            sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=int(num_sampled_clients*self.fraction), replace=False).tolist())
            for idx in sampled_client_indices:
                self.hired_px_hist.append(self.clients[idx].get_asking_price())
                
        return sampled_client_indices
    
    
    """Step 2: Selected Client starts training"""
    def update_selected_clients(self, sampled_client_indices):
        
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    """Step 3: Evaluate the clients and update their reputation"""
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        #update their reputation and most recent performance
        if self.incentive_type == 'oort':
            #update and clip feedback and blacklist outliers
            for client_id in sampled_client_indices:
                test_loss, test_accuracy = self.clients[client_id].client_evaluate()
                #count will be added in update function
                feedbacks = {'reward': test_accuracy, 'duration':self.clients[client_id].get_time_taken(), 'time_stamp':self._round, 'count': self.clients[client_id].get_num_selected(), 'status': True}
                self.oort_train_selector.update_client_util(client_id, feedbacks)
        else:
            for idx in sampled_client_indices:
                test_loss, test_accuracy = self.clients[idx].client_evaluate()
                if self.incentive_type == "RRAFL" and self._round >= 3:
                    self.update_select_client_rep_RRAFL(idx, test_accuracy)
                #else use normal rep update
                self.update_select_client_rep(idx,test_accuracy)
            #update average reputation value of selected clients
            self.avg_rep = sum([self.client_rep_list[idx][0] for idx in sampled_client_indices])/len(sampled_client_indices)

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    """Update reputation based on BRS"""
    def update_select_client_rep(self, idx, test_accuracy ):
        if idx not in self.client_rep_list:
            #[reputation value, alpha, beta, acc] 
            self.client_rep_list[idx] = [0.5, 0 , 0 , test_accuracy]
            return

        if test_accuracy > self.client_rep_list[idx][3]:
            self.client_rep_list[idx][1] += (2**(self.num_rounds-self._round)) * 1
        else: 
            self.client_rep_list[idx][2] += (2**(self.num_rounds-self._round)) * 1

        new_rep = (self.client_rep_list[idx][1] + 1) /  (self.client_rep_list[idx][2] + self.client_rep_list[idx][1] + 2)       
        self.client_rep_list[idx][0] = new_rep
        self.client_rep_list[idx][3] = test_accuracy
    
    def update_select_client_rep_RRAFL(self, idx, test_accuracy):
        #Do contribution measurement first
        curr_grad = self.clients[idx].get_grad() 
        client_contrib =  calculate_client_contrib(curr_grad, self.global_grad)
        self.clients[idx].update_contrib_list(client_contrib)
        max_contrib = max(self.clients[idx].get_contrib_list())
        z =  max(0.001, client_contrib) / max_contrib
        #calculate if client should get selected or not, with loss threshold = -0.01 as set in the paper
        if  self.client_rep_list[idx][3] - test_accuracy >= -0.01:
            self.clients[idx].inc_num_pass()
        else:
            self.clients[idx].inc_num_fail()
        #gomertz function for incentive calculation
        x = (0.4* self.clients[idx].get_num_pass())-(0.6*self.clients[idx].get_num_fail()) /  (0.4* self.clients[idx].get_num_pass())+(0.6*self.clients[idx].get_num_fail())
        y = math.pow(math.e, -1*math.pow(math.e, -5.5*x))
        new_rep =  y*z

        self.client_rep_list[idx][0] = new_rep

   
    #-----------------UTILITY FUNCTIONS----------------------------
    def cal_avg_price(self, rounds=None):
        if len(self.hired_px_hist) == 0:
            return 0
        if rounds == None:
            return (sum(self.hired_px_hist)/len(self.hired_px_hist))
        else:
            return (sum(self.hired_px_hist[:-1])/(len(self.hired_px_hist)-1))

    def calculate_utility(self, acc):
        utility = (self.multiplier_g * acc * 100)-self.used_PR_budget
        return utility
    
    #call this action every round to update per round budget
    def set_budget(self):
        self.PR_used_budget_hist.append(self.used_PR_budget)
        print(f"per round used budget {self.used_PR_budget}")
        #set incentive type
        if (self._Q) > 0  :
            self.incentive_type = "ReverseMod"
            # if self.l2norm != 0:
            remaining_budget = self.PR_budget - self.used_PR_budget
            print(f"remaining budget {remaining_budget}")
            self.PR_budget = remaining_budget + self.basic_PR_budget
            self.hire_budget = (self._Q)+(self.opt_bid)-(self._V)
            #Put a cap on hire budget, cannot be more than Per round budget
            if self.hire_budget > self.PR_budget:
                self.hire_budget = self.PR_budget
            if self.hire_budget < 0:
                self.hire_budget = 0
            print(f'hire budget for round {self._round} is {self.hire_budget}')
        else:
            self.incentive_type = "Reverse"
            remaining_budget = self.PR_budget-self.used_PR_budget
            print(f"remaining budget {remaining_budget}")
            self.PR_budget = remaining_budget + self.basic_PR_budget
            print(f'per round budget for round {self._round} is {self.PR_budget}')
        
    def get_optimal_bid(self, clients_bids):
        temp = 0
        for key in clients_bids:
            if key in self.client_rep_list.keys():
                if self.client_rep_list[key][0] >= self.rep_threshold:
                    temp += clients_bids[key]
        return temp

        
    def set_overall_l2norm(self, l2norm):
        self.l2norm = l2norm

    def get_PR_budget_hist(self):
        return self.PR_used_budget_hist

    def get_results(self):
        return self.results
    
    def get_Q_hist(self):
        return self._Q_hist

    def calculate_Q(self, acc):
        utility = self.calculate_utility(acc)
        self.utility_history.append(utility)
        print(f"Utility {utility}")
        
        optimal_improvement_threshold = get_optimal_improvement_threshold(self._round)
        if self._round >= 1:
            actual_improvement = acc - self.results['accuracy'][self._round-1]
        else:
            actual_improvement = 0
        print(f"optimal improvement is {optimal_improvement_threshold} while current rate is {actual_improvement}")
        if (actual_improvement <= optimal_improvement_threshold ):
            self._Q = max(0, self._Q + self.opt_bid - self.used_PR_budget)
        else:
            self._Q = max(0, self._Q - self.used_PR_budget)
        print(f'Result Q = {self._Q}')
        self._Q_hist.append(self._Q)
        
    def get_q_square(self):
       q_f_2 = self._Q * self._Q 
       return q_f_2

    def save_model(self):
        PATH = os.getcwd()
        PATHNAME = os.path.join(PATH,'model.pth')
        torch.save(self.model, PATHNAME)
    
    def get_Q(self):
        return self._Q

    def get_utility_history(self):
        return self.utility_history
    
    def calculate_client_jfi(self):
        jfi = calculate_JFI_clients(self.clients, self.client_rep_list)
        print(f"JFI for {self.incentive_type} is {jfi}")
   
    def prune_client_tasks(self, clientSampler, sampledClientsRealTemp, numToRealRun, global_virtual_clock):

        sampledClientsReal = []
        # 1. remove dummy clients that are not available to the end of training
        for virtualClient in sampledClientsRealTemp:
            roundDuration = clientSampler.getCompletionTime(virtualClient,
                                    batch_size=self.batch_size, upload_epoch=self.local_epochs,
                                    model_size=65556) * self.clock_factor

            if clientSampler.isClientActive(virtualClient, roundDuration + global_virtual_clock):
                sampledClientsReal.append(virtualClient)

        # 2. we decide to simulate the wall time and remove 1. stragglers 2. off-line
        completionTimes = []
        virtual_client_clock = {}
        for virtualClient in sampledClientsReal:
            roundDuration = clientSampler.getCompletionTime(virtualClient,
                                    batch_size=self.batch_size, upload_epoch=self.upload_epoch,
                                    model_size=self.model_size) * self.clock_factor
            completionTimes.append(roundDuration)
            virtual_client_clock[virtualClient] = roundDuration

        # 3. get the top-k completions
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        top_k_index = sortedWorkersByCompletion[:numToRealRun]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[numToRealRun:]]
        round_duration = completionTimes[top_k_index[-1]]
        
        return clients_to_run, dummy_clients, virtual_client_clock, round_duration

    
    #---------------------MAIN SERVER FUNCTIONS-------------------------
    #-------------------MODEL RELATED FUNCTIONS-------------------------
    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(len(self.clients))} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            # assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()


    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        all_client_grad = []
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if self.incentive_type == "FIFL":
                all_client_grad.append(self.clients[idx].get_grad())
            #time sensitive
            if self.incentive_type == "Reverse" or self.incentive_type == "ReverseMod":
                time_taken_client = self.clients[idx].get_time_taken()
                if time_taken_client > self.time_required:
                    #get back the money spent
                    self.used_PR_budget -= self.clients[idx].get_asking_price()
                    #skip aggregating for this client if too slow
                    continue
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights, strict=False)
        if self.incentive_type == "FIFL":
            self.global_grad = np.mean(all_client_grad)
            print(f"gradient of server is {self.global_grad}")
        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect() 

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device, non_blocking=True)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device, non_blocking=True), labels.long().to(self.device, non_blocking=True)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device == "cuda": torch.cuda.empty_cache()
        # self.model.to("cpu")
        
        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy
    
    def train_federated_model(self, client_visit_arr, clients_bids):
        """Do federated training."""
        #reset PR budget
        self.used_PR_budget = 0
        
    
        sampled_client_indices = self.sample_clients(client_visit_arr, clients_bids)       
        self.hire_count_now = len(sampled_client_indices)


        print(f'id of selected clients: {sampled_client_indices}')
        #Clients can choose to accept or not
    
        # send global model to the selected clients, \omega_n^t = \omega_m^t
        self.transmit_model(sampled_client_indices)
        
        if len(sampled_client_indices) == 0:
            return sampled_client_indices

        # train and update model for selected clients with local dataset
        selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        self.evaluate_selected_models(sampled_client_indices)

        # print(self.client_rep_list)

        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)

        #free the client after training
        for idx in sampled_client_indices:
            self.clients[idx].set_available()
        
   
            
    def fit(self, r, client_visit_arr, clients_bids):
        """Execute the whole process of the federated learning."""

        self._round = r            
        # self.total_bid = sum(clients_bids.values())
        # print("total bid is"+ str(self.total_bid))

        self.opt_bid = self.get_optimal_bid(clients_bids)
        print("optimal bid is"+ str(self.opt_bid))

        self.train_federated_model(client_visit_arr, clients_bids)
        test_loss, test_accuracy = self.evaluate_global_model()

        
        self.results['loss'].append(test_loss)
        self.results['accuracy'].append(test_accuracy)
    
        self.writer.add_scalars(
            'Loss',
            {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
            self._round
            )
        self.writer.add_scalars(
            'Accuracy', 
            {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
            self._round
            )

        message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
            \n\t[Server: {str(self.id) }] ...finished evaluation!\
            \n\t=> Loss: {test_loss:.4f}\
            \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
        print(message); logging.info(message)
        del message; gc.collect()
        
        self.transmit_model()
        #Calculate Q and Calculate the utility yield
        self.calculate_Q(test_accuracy)
        #ONLY FOR GPSS-ALGO
        if self.incentive_type == "Reverse" or self.incentive_type == "ReverseMod":
            self.set_budget()
        else:
            self.PR_used_budget_hist.append(self.used_PR_budget)
        print(f'PR_budget for Fed {self.id} is {self.PR_budget}')
        
    
   