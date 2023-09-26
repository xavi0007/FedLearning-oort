import copy
import itertools
import random
import numpy as np

def calculate_reward(rep, threshold_b ,client_grad,global_gradient, client_visit_arr, clients_list):
    grad_dist = calculate_gradient_distance(client_grad, global_gradient)
    incentive = rep *  ( ((threshold_b-grad_dist)/threshold_b) / calculate_all_contribution(clients_list, client_visit_arr, global_gradient) )
    # print(incentive)
    return incentive

def calculate_all_contribution(clients_list, client_visit_arr, global_gradient):
    all_contrib = []
    for client_id in client_visit_arr:
        if clients_list[client_id].get_grad() != None:
            all_contrib.append(calculate_gradient_distance(clients_list[client_id].get_grad(), global_gradient))
    return sum(all_contrib)

def calculate_gradient_distance(client_grad, global_gradient):
    dist = np.linalg.norm(global_gradient - client_grad)
    return dist


