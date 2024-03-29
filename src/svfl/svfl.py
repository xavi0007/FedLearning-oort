import copy
import itertools
import random

def calculate_sv_old(models, model_evaluation_func, averaging_func):
    """
    Computes the Shapley Value for clients

    Parameters:
    models (dict): Key value pair of client identifiers and model updates.
    model_evaluation_func (func) : Function to evaluate model update.
    averaging_func (func) : Function to used to average the model updates.

    Returns:
    sv: Key value pair of client identifiers and the computed shapley values.

    """

    # generate possible permutations
    all_perms = list(itertools.permutations(list(models.keys())))
    marginal_contributions = []
    # history map to avoid retesting the models
    history = {}

    for perm in all_perms:
        perm_values = {}
        local_models = {}

        for client_id in perm:
            model = copy.deepcopy(models[client_id])
            local_models[client_id] = model

            # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
            if len(perm_values.keys()) == 0:
                index = (client_id,)
            else:
                index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

            if index in history.keys():
                current_value = history[index]
            else:
                model = averaging_func(local_models)
                current_value = model_evaluation_func(model)
                history[index] = current_value

            perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

        marginal_contributions.append(perm_values)

    sv = {client_id: 0 for client_id in models.keys()}

    # sum the marginal contributions
    for perm in marginal_contributions:
        for key, value in perm.items():
            sv[key] += value

    # compute the average marginal contribution
    sv = {key: value / len(marginal_contributions) for key, value in sv.items()}

    return sv



def calculate_sv(client_visit_arr, client_rep_list):

    # generate possible permutations
    # all_perms = list(itertools.permutations(client_visit_arr))
    all_perms=[]
    #Randomly generate a set number of perms
    # Data Shapley: Equitable Valuation of Data for Machine Learning.
    
    for _ in range( int(len(client_visit_arr)*0.5)):
        permuted_items = random.sample(client_visit_arr, k=int(len(client_visit_arr)*0.2))
        all_perms.append(permuted_items)



    marginal_contributions = []
    # history map to avoid retesting the models
    history = {}
    for perm in all_perms:
        perm_values = {}

        for client_id in perm:
            # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
            if len(perm_values.keys()) == 0:
                index = (client_id,)
            else:
                index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

            if index in history.keys():
                current_value = history[index]
            else:
                #current = most recent acc
                current_value = client_rep_list[client_id][3]
                history[index] = current_value

            perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

        marginal_contributions.append(perm_values)

    sv = {client_id: 0 for client_id in client_visit_arr}
     # sum the marginal contributions
    for perm in marginal_contributions:
        for key, value in perm.items():
            sv[key] += value

    # compute the average marginal contribution
    
    sv = {key: value / len(marginal_contributions) for key, value in sv.items()}

    print(sv)

    return sv
