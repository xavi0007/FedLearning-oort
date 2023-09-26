import tqdm
from copy import deepcopy
import gc
import logging
import time
import torch
import random
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device, cost):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.available = True
        self.data = local_data
        self.device = device
        self.__model = None
        self.cost = cost
        #Randomised
        self.type = "Market"
        self.asking_price = 0.8
        self.grad = 0
        self.number_of_accepted = 0
        self.time_taken = 0
        self.contrib_list = []
        self.num_pass = 1
        self.num_fail = 1

        self.compute_speed = 1
        self.bandwidth = 1
        self.distance = 1
        self.size = 1
        self.score = 1
        self.traces = 1
        self.behavior_index = 0

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model
    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model
    
    def get_num_pass(self):
        return self.num_pass
    def get_num_fail(self):
        return self.num_fail
    
    def inc_num_pass(self):
        self.num_pass += 1
    
    def inc_num_fail(self):
        self.num_fail += 1

    def get_id(self):
        return self.id 

    def get_grad(self):
        return self.grad

    def get_asking_price(self):
         return self.asking_price

    def set_available(self):
        self.available = True

    def get_availabe(self):
        return self.available

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)
    
    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
    
    def getScore(self):
        return self.score

    def registerReward(self, reward):
        self.score = reward

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

        #Starting price
        self.asking_price = random.uniform(self.asking_price, 3)
        self.algo_type = None

    def set_available(self):
        self.available = True
    def update_contrib_list(self, current_contrib):
        self.contrib_list.append(current_contrib)
    def get_contrib_list(self):
        return self.contrib_list
    def set_type(self, type):
        self.type = type
    def get_num_selected(self):
        return self.number_of_accepted
    def send_message(self, msg):
        # self.last_avg_price =  msg[1]
        # self.max_price = msg[2]
        if msg[0] == "Accept":
            self.number_of_accepted += 1
            self.available = False
            if self.type == 'Market':
                # self.asking_price = max(self.total_cost, 1.2*msg[1])
                self.asking_price = max(msg[2], self.asking_price)
                # self.asking_price = random.uniform(self.asking_price, msg[2])
            elif self.type == "Randomised":
                self.asking_price = random.uniform(self.asking_price, msg[2])
        elif msg[0] == "Decline":
            self.available = True
            if self.type == 'Market':
                # self.asking_price = min(msg[2], self.asking_price)
                self.asking_price = random.uniform(self.asking_price, msg[2])
            # elif self.type == "Randomised":
            #     self.asking_price = random.uniform(self.asking_price, msg[2])

    def set_algo_type(self, algo_type):
        self.type = algo_type

    def get_time_taken(self):
        return self.time_taken
        
    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device, non_blocking=True)    
        start = time.time()
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        for _ in range(self.local_epoch):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device, non_blocking=True), labels.long().to(self.device,non_blocking=True)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = eval(self.criterion)()(outputs, labels)    
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step() 

                if self.device == "cuda": torch.cuda.empty_cache()       
        # scheduler.step()        
        count = 0
        sum_grad = 0
        if self.algo_type == "FIFL" or self.algo_type == "RRAFL":
            for param in self.model.parameters():
                sum_grad += param.grad.mean().item()
                count += param.numel()
            self.grad = sum_grad / count
            self.grad = sum_grad
        # print(f"gradient of client is {self.grad}")
        end = time.time()
        self.time_taken = end - start
        # print(self.time_taken)
        msg = f"it took the base client to execute this number of epochs: { self.local_epoch}"
        # print(msg); logging.info(msg)
    
    def getCompletionTime(self, batch_size, upload_epoch, model_size):
        return (3.0 * batch_size * upload_epoch/float(self.compute_speed) + model_size/float(self.bandwidth))
        
    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device, non_blocking=True)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device,non_blocking=True), labels.long().to(self.device, non_blocking=True)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                if self.device == "cuda": torch.cuda.empty_cache()
        # self.model.to(self.device)
        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        
        
        return test_loss, test_accuracy

