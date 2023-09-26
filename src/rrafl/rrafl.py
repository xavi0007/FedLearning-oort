import copy
import itertools
import random
import numpy as np
import math



# def calculate_max_contribution(clients_list, client_visit_arr, global_gradient):
#     all_contrib = []
#     for client_id in client_visit_arr:
#         if clients_list[client_id].get_grad() != None:
#             all_contrib.append(calculate_client_contrib(clients_list[client_id].get_grad(), global_gradient))
#     return max(all_contrib)

# def calculate_all_contribution(clients_list, client_visit_arr, global_gradient):
#     all_contrib = []
#     for client_id in client_visit_arr:
#         if clients_list[client_id].get_grad() != None:
#             all_contrib.append(calculate_client_contrib(clients_list[client_id].get_grad(), global_gradient))
#     return sum(all_contrib)

def calculate_client_contrib(client_grad, global_gradient):
    if client_grad == None:
        client_grad = 0
    if global_gradient == None:
        global_gradient = 0
    dist = global_gradient*np.cos(client_grad)* abs(np.cos(client_grad))
    return dist
