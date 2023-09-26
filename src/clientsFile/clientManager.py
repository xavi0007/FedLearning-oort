
import gc
import logging
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from src.models import *
from src.utils import *
from src.dataFile.datapipeline import create_datasets
from .client import Client
from .badclient import BadClient

logger = logging.getLogger(__name__)

class ClientManager(object):
    def __init__(self,  writer, device,  data_config={}, fed_config={}, optim_config={}, cost=0):
        self.clients = []
        self.writer = writer
        self.device = device
        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]       

        self.local_epochs = fed_config["E"] #number of epochs in device
        self.batch_size = fed_config["B"] #batch size  
        self.num_clients = fed_config["K"] #number of clients
        
        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]

        self.optim_config = optim_config
        self.cost_per_unit_data = cost
        self.dataloader = None
        self.idx_arr = []

    def setup(self):
          # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)

        # assign dataset to each client
        self.clients = self.create_good_clients_only(local_datasets)
        # self.clients = self.create_clients(local_datasets, 15)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
    def get_visit_arr(self,):
        return self.idx_arr
    def randomize_visits(self, num_fed=1):
        #shuffle client idx list
        client_visit_lst =  [x for x in random.sample(self.idx_arr, len(self.idx_arr))]
        if num_fed == 1:
            return client_visit_lst
        else:
            length = len(client_visit_lst)
            parts = [client_visit_lst[i*length//num_fed:(i+1)*length//num_fed] for i in range(num_fed)]
            #returns an array of client idx
            return parts

    def get_client_bids(self):
        bids = {}
        for client in self.clients:
            bids[client.get_id()] = client.get_asking_price()
        return bids
        
    def create_good_clients_only(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device, cost=self.cost_per_unit_data)
            clients.append(client)
            self.idx_arr.append(k)
            #reputation of all clients start with 0.5 rep and 0 accuracy

            
        message = f"[...successfully created all {str(len(clients))} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def set_num_clients(self, number_of_client):
        self.num_clients = number_of_client
    
    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[...successfully finished setup of all {str(len(self.clients))} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def set_client_type(self, type):
        for client in self.clients:
            client.set_type(type)
        

    def get_clients(self):
        return self.clients
    
    def get_evaluate_dataloader(self):
        return self.data , self.dataloader

    def create_clients(self, local_datasets, bad_client_count):
        """Initialize each Client instance."""
        clients = []
        assert(self.num_clients > bad_client_count)
        count = 1
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            if count < bad_client_count:
                clients.append(BadClient(client_id=k, local_data=dataset, device=self.device, cost=self.cost_per_unit_data, attack_type='label_flip'))
                count+=1
            else:
                clients.append(Client( client_id=k, local_data=dataset, device=self.device, cost=self.cost_per_unit_data))
        
        message = f"[...successfully created all {str(len(clients))} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

        return clients