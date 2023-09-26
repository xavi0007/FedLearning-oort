import tqdm
from copy import deepcopy
import gc
import logging
import time
import torch
import random
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BadClient(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device, cost ,attack_type):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.available = True
        self.data = local_data
        self.device = device
        self.__model = None
        self.cost = cost
        #dictionary; idx = fed_id,  value = (alpha , beta, rep)
        self.Fed_rep = {}
        self.threshold = 0.7
        self.last_offered = 0
        self.reject_counter = 0
        self.attack_type = attack_type
    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)
    
    
    
    def set_reward(self, _reward):
        self.reward = _reward
    
    def cal_brs(self, utility, fed_id):
        alpha , beta = 0 , 0 
        if utility > (self.cost*len(self.data)):
            alpha = 1
        else: 
            beta = 1
        if fed_id in self.Fed_rep:
            #new alpha and beta
            self.Fed_rep[fed_id][0] += alpha 
            self.Fed_rep[fed_id][1] += beta 
            new_rep =  (self.Fed_rep[fed_id][0] + 1) /  (self.Fed_rep[fed_id][1] + self.Fed_rep[fed_id][0] + 2)            
            self.Fed_rep[fed_id][2] = new_rep
        else:
            self.Fed_rep[fed_id] = [alpha, beta, 0.5] 

    def calculate_utility(self, _reward):
        utility = _reward - (self.cost*(len(self.data)))
        return utility
        
        

    #randomly accept or check reputation., depend on strategy
    def acceptance(self, fed_id,  _reward, _round):
  
        if (self.available == False):
            return False

        utility = self.calculate_utility( _reward)        
        self.cal_brs(utility, fed_id)
        self.available = True
        return True

        
    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.optim_config = client_config["optim_config"]

    def set_available(self):
        self.available = True

    def client_update(self):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device, non_blocking=True)
        optimizer = eval(self.optimizer)(self.model.parameters(), **self.optim_config)
        if self.attack_type == 'free_rider':
            adv_grad = []
            for param in self.model.parameters():
                adv_grad.append((torch.rand(param.size()) * 2 - 1).to(self.device))
        elif self.attack_type == 'label_flip':
            for _ in range(self.local_epoch):
                #100% perform this attack
                chance = 10
                randomnum = random.randrange(0,9)
                if randomnum < chance:
                    for data, labels in self.dataloader:
                        for idx, label in enumerate(labels):
                            if int(label) == 7:
                                labels[idx] = 1
                            elif int(label) == 1:
                                labels[idx] = 7
                            data, labels = data.float().to(self.device, non_blocking=True), labels.long().to(self.device,non_blocking=True)
                            
                            optimizer.zero_grad()
                            outputs = self.model(data)
                            loss = eval(self.criterion)()(outputs, labels)    
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                            optimizer.step() 

                            if self.device == "cuda": torch.cuda.empty_cache()               
       
        
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

