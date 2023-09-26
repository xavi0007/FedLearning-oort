import os
import logging
import copy
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math  
  
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, device, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    
    if device == 'cuda' :
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            model.to(gpu_ids[0])
            model = nn.DataParallel(model, gpu_ids)
    else:
        model.to(device)
        
    init_weights(model, init_type, init_gain)
    return model

def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy > best_accuracy

def compute_accuracy(self):
    # compute accuracy
    correct = torch.Tensor([0])
    for i in range(len(self.predictions_list)):
        all_preds = self.predictions_list[i]
        pred = all_preds.data.max(1, keepdim=True)[1]

        assert pred.device == self.targets_list[i].device, (
            f"Pred and targets moved to different devices: "
            f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
        )
        if i == 0:
            correct = correct.to(pred.device)

        correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

    # total number of data
    total = sum(len(batch_targets) for batch_targets in self.targets_list)

    accuracy = 100.0 * correct.item() / total
    return accuracy

def compute_grad_update(old_model, new_model,  device):
    old_model , new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
	assert len(grad_update_1) == len(
		grad_update_2), "Lengths of the two grad_updates not equal"
	
	for param_1, param_2 in zip(grad_update_1, grad_update_2):
		param_1.data += param_2.data * weight

def sign(grad):
	return [torch.sign(update) for update in grad]

def flatten(grad_update):
	return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
	grad_update = []
	for param in normal_shape:
		n_params = len(param.view(-1))
		grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
		flattened = flattened[n_params:]

	return grad_update

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
def l2norm(grad):
	return torch.sqrt(torch.sum(torch.pow(flatten(grad), 2)))

def cosine_similarity(grad1, grad2, normalized=False):
	"""
	Input: two sets of gradients of the same shape
	Output range: [-1, 1]
	"""

	cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10) 
	if normalized:
		return (cos_sim + 1) / 2.0
	else:
		return cos_sim

def angular_similarity(grad1, grad2):
	return 1 - torch.div(torch.acovs(cosine_similarity(grad1, grad2)), pi)

def add_update_to_model(model, update, weight=1.0, device=None):
	if not update: return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model

def compare_models(model1, model2):
	for p1, p2 in zip(model1.parameters(), model2.parameters()):
		if p1.data.ne(p2.data).sum() > 0:
			return False # two models have different weights
	return True

def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile is not None:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
            #topk returns the k largest elements from the given tensor
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		mask_percentile = max(0, mask_percentile)
		for i, layer in enumerate(grad_update):
			layer_mod = layer.data.view(-1).abs()
			if mask_percentile is not None:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))																																												
				grad_update[i].data[layer.data.abs() < topk[-1]] = 0
		return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update

#obsolete 
# def get_optimal_util(round, multiplier):
#     pwd = os.getcwd()
#     fp = os.path.join(pwd,'log', 'result.pkl')    
#     benchmark = pd.read_pickle("/Users/xavier/Programming/FedLearning-main/log/Benchmark_MNIST/result.pkl")

#     acc = benchmark['accuracy'][round]
#     revenue = 100 * acc * multiplier
#     #minimal cost , 0.0001 cost of client per data * 1500 number of data * 0.5*40 clients
#     #min asking price = 1  * 0.5*40
#     if  (round > 60):
#         cost = 1*15
#     elif (round > 80):
#         cost = 10
#     else:
#         cost = 1*18
#     utility = revenue - cost

#     return utility

def get_optimal_improvement_threshold(round):
    pwd = os.getcwd()
    fp = os.path.join(pwd,'log', 'result.pkl')    
    # benchmark = pd.read_pickle("/Users/xavier/Programming/FedLearning-main/log/Benchmark_K60_R80_B400_MNIST_reverse/result.pkl")
    benchmark = pd.read_pickle("/Users/xavier/Programming/FedLearning-main/log/Benchmark_EMNIST_K20/result.pkl")
    acc = benchmark['accuracy'][round]
    prev_acc = benchmark['accuracy'][round-1] 
    optimal_improvement = acc - prev_acc 
    # if optimal_improvement < 0:
    #     optimal_improvement += 0.001
    # else:
    #     optimal_improvement += 0.0008
    return optimal_improvement
      

def calculate_prob_selected(number_selected, rounds=60):
    return (number_selected/rounds) * 100

def calculate_JFI_clients(clients, client_rep_list):
    arr = []
    arr2 = []
    for idx in range(len(clients)):
        client_rep_list
        num_selected = clients[idx].get_num_selected()
        if(num_selected == 0):
            continue
        # prob = calculate_prob_selected(num_selected) 
        # rep = client[0]
        if(client_rep_list[idx][0] <= 0):
            continue
        frac = calculate_prob_selected(num_selected) /client_rep_list[idx][0]
        arr.append(frac)
        arr2.append(frac*frac)
    
    top =  sum(arr)*sum(arr) 
    bottom = sum(arr2)
    jfi = top / (len(clients) * bottom)
    return jfi

def calculate_l(q_dict, ex_key):
    other_q_square = []
    for key in q_dict:
        if key == ex_key:
            continue
        other_q_square.append(q_dict[key])        
    return math.sqrt(sum(other_q_square)) * 0.5
#--------------------------Graph---------------------------------------
def graph_results(results: dict):
    #results = {"loss": [], "accuracy": []}
    performance_arr = results["accuracy"]
    x_axis = [x for x in range(len(performance_arr))]
    fig, ax1 = plt.subplots(dpi = 300)
    ax1.plot(x_axis,performance_arr, label='Budget for Fed f', color='blue', marker='x')
    ax1.set_ylabel('Performance', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Time frame, T", fontsize=15,  fontweight='bold')
    
      
    ax1.legend(loc=1, fontsize= 15)
    plt.grid()
    plt.savefig("util.pdf", dpi = 300)

def graph_budget(arr):
    x_axis = [x for x in range(len(arr))]
    fig, ax1 = plt.subplots(dpi = 300)
    ax1.plot(x_axis,arr, label='Budget for Fed 1', color='blue', marker='x')
    ax1.set_ylabel('budget', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
      
    ax1.legend(loc=1, fontsize= 15)
    plt.grid()
    plt.savefig("budget.pdf", dpi = 300)


def graph_util(arr):
    x_axis = [x for x in range(len(arr))]
    fig, ax1 = plt.subplots(dpi = 300)
    ax1.plot(x_axis,arr, label='Budget for Fed 1', color='blue', marker='x')
    ax1.set_ylabel('Utility', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
      
    ax1.legend(loc=1, fontsize= 15)
    plt.grid()
    plt.savefig("util.pdf", dpi = 300)


def graph_Q(Q_hist):

    x_axis = [x for x in range(len(Q_hist))]
    
    fig, ax1 = plt.subplots(dpi = 300)

    
    ax1.plot(x_axis,Q_hist, label='Q value for Fed 1', color='blue', marker='o')    
    ax1.set_ylabel('Q', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
    
    ax1.legend(loc=1, fontsize= 15)
    # ax2.legend(loc=1, fontsize= 15)
    # plt.show()
    plt.grid()
    plt.savefig("Q_value.pdf", dpi = 300)

def graph_many_Q(Q_hist_1, Q_hist_2, Q_hist_3):

    x_axis = [x for x in range(len(Q_hist_1))]
    
    fig, ax1 = plt.subplots(dpi = 500)

    
    ax1.plot(x_axis, Q_hist_1, label='Fed 1', color='red', linestyle='dashdot')    
    ax1.plot(x_axis, Q_hist_2, label='Fed 2', color='green', linestyle='dashed')    
    ax1.plot(x_axis, Q_hist_3, label='Fed 3', color='blue')    
    ax1.set_ylabel('Q', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
    
    ax1.legend(loc=1, fontsize= 15)
    # ax2.legend(loc=1, fontsize= 15)
    # plt.show()
    plt.grid()
    plt.savefig("all_Q_value.pdf", dpi = 500)

def graph_many_utility(util_hist_1, util_hist_2, util_hist_3):
    print("Market Utils:")
    print(util_hist_1)
    print("Random Utils:")
    print(util_hist_2)
    print("Cost Utils:")
    print(util_hist_3)
    x_axis = [x for x in range(len(util_hist_1))]
    
    fig, ax1 = plt.subplots(dpi = 500)

    # fed1 = [29.65, 34.81, 25.112499999999994, 36.10524999999999, 47.94639441747573, 51.24999999999999, 50.849999999999994, 50.870000000000005, 51.4, 50.160000000000004, 51.41, 51.370000000000005, 51.01, 50.24999999999999, 50.849999999999994, 50.43, 49.830000000000005, 50.74999999999999, 50.79, 50.5, 50.6, 50.03999999999999, 51.559999999999995, 50.67, 50.22, 50.690000000000005, 50.38, 50.79, 50.28, 50.480000000000004, 51.18000000000001, 50.18, 50.57000000000001, 51.629999999999995, 50.7, 49.87, 50.96000000000001, 50.81, 50.470000000000006, 49.97, 50.56, 51.800000000000004, 50.71, 50.99, 50.370000000000005, 50.74999999999999, 50.56, 50.849999999999994, 49.3, 49.75, 50.56, 50.31, 46.01, 53.1485, 55.47, 55.36, 55.50000000000001, 55.16]
    
    ax1.plot(x_axis, util_hist_1, label='Market', color='red', linestyle='dashdot')    
    ax1.plot(x_axis, util_hist_2, label='Random', color='green', linestyle='dashed')    
    ax1.plot(x_axis, util_hist_3, label='Cost', color='blue')    
    ax1.set_ylabel('Utility', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
    
    ax1.legend(fontsize= 15)
    # ax2.legend(loc=1, fontsize= 15)
    # plt.show()
    plt.grid()
    plt.savefig("all_util_value.pdf", dpi = 500)
# graph_many_utility([33.1, 42.03, 42.17, 54.15, 57.92000000000001, 61.290000000000006, 65.74, 67.33, 68.78, 73.02, 73.28, 76.08000000000001, 77.08000000000001, 77.65, 79.99, 74.64999999999999, 77.97, 81.37, 81.35000000000001, 81.55, 82.21, 75.67, 72.71, 76.79, 77.69000000000001, 82.46000000000001, 87.14, 83.28, 83.5, 67.26, 75.12, 64.47, 81.31, 82.09, 82.6, 72.25, 83.7, 88.85, 84.17, 88.31, 86.87, 81.72000000000001, 86.97000000000001, 89.74, 89.82000000000001, 88.77000000000001, 90.62, 89.7, 90.31, 90.15, 77.92, 90.21000000000001, 90.83, 90.55, 76.43, 82.13000000000001, 83.37, 88.14999999999999], [29.759999999999998, 36.28, 30.630000000000003, 43.69594568541675, 53.110764758533094, 51.03341924636312, 47.58230654051328, 61.765289576630344, 65.94462883651615, 67.80595886804372, 71.90001767522722, 68.69279269905216, 74.10838269090799, 76.51324187578936, 76.43225116966744, 78.60986542132828, 79.33105197357627, 73.55031154964915, 79.95685825693486, 78.2271512167085, 73.14629626916285, 83.30938286481495, 81.08570791680648, 82.55635845561407, 84.62872227073032, 81.96406130876481, 85.72437015604481, 84.23667433811444, 67.06846378804107, 87.23099070379115, 85.03951498848228, 87.13675806797367, 88.35709056689048, 87.89147202187453, 87.80968435373218, 83.82321469098501, 87.29782471801454, 87.30376076537281, 88.98402774969449, 89.18104688101042, 81.67573057183614, 79.61761711825667, 81.9825428367458, 85.752601605693, 90.24426910983931, 78.38930895390948, 86.50133226169297, 90.24737889365504, 83.62395538604011, 83.52043001508639, 83.27494088633809, 90.9329471551794, 83.37092323241531, 89.1924467884871, 88.96972398973145, 88.07229711115454, 88.09673303286927, 91.1168931367016], [35.13, 44.04, 46.35999999999999, 45.07, 58.580000000000005, 54.67, 55.00000000000001, 70.61, 62.190000000000005, 74.07000000000001, 72.11, 71.79, 74.58, 76.26, 74.36000000000001, 80.31, 78.85000000000001, 80.76, 81.23, 80.77, 81.24000000000001, 75.09, 84.05000000000001, 85.08, 85.03, 77.35, 86.35000000000001, 86.41, 83.42, 85.55000000000001, 87.08, 86.94, 87.38000000000001, 88.35, 87.14, 86.88000000000001, 81.53, 75.35, 89.04, 88.42, 88.12, 88.66, 88.86, 89.01, 89.47, 69.8, 88.64, 80.74000000000001, 88.93, 74.66000000000001, 89.09, 89.85, 89.63, 90.61, 87.85, 89.44, 89.80000000000001, 89.80000000000001])

def graph_many_budget( budget_hist_1, budget_hist_2, budget_hist_3):

    print("Market budget:")
    print(budget_hist_1)
    print("Random budget:")
    print(budget_hist_2)
    print("Cost budget:")
    print(budget_hist_3)

    x_axis = [x for x in range(len(budget_hist_1))]
    
    fig, ax1 = plt.subplots(dpi = 500)
    # budget_hist_1 = [0, 72.12, 65.56, 92.49, 29.277499999999957, 14.131999999999923, 0.16235334582441197, 0.11556968235969833, 41.146363746856395, 1.8495366365402188, 0.6283232569955084, 5.993657040698322, 1.9483269362309894, 0.013644679289244621, 0.758975353086345, 0.21117339597409623, 0.4983826983953872, 0.5039755296886645, 0.40071093261331026, 0.22228364863019623, 0.9583580412269883, 0.3045180590050407, 0.9302101787196473, 0.902729430044507, 0.5471476975561771, 0.5278526734199471, 0.22300925501055202, 0.2956901466670274, 0.2373936323558472, 0.2431253477552402, 1.0129329795722577, 1.0914474740594788, 0.615914387085152, 0.9566832024460599, 1, 0.025311594082947364, 1.0300000000000011, 0.5735066083643834, 1.006225702553458, 1, 0.7668922601143664, 0.02141912873925156, 0.48196568638665305, 1.065021233134221, 1, 0.5335945566988891, 0.5106732895124608, 1, 1.019999999999996, 1, 1, 1, 1.0500000000000114, 1, 1, 1, 0.8768009969343571, 0.0793267042482424, 1, 0.7594329613004109]
    
    ax1.plot(x_axis, budget_hist_1, label='Market', color='red', linestyle='dashdot')    
    ax1.plot(x_axis, budget_hist_2, label='Random', color='green', linestyle='dashed')    
    ax1.plot(x_axis, budget_hist_3, label='Cost', color='blue')    
    ax1.set_ylabel('Budget', fontsize=15, fontweight='bold')
    # ax2.set_ylabel('Overall cost', fontsize=15)
    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
    
    ax1.legend(loc=1, fontsize= 15)
    # ax2.legend(loc=1, fontsize= 15)
    # plt.show()
    plt.grid()
    plt.savefig("all_B_value.pdf", dpi = 500)

def graph_combine(id, budget_arr, Q_arr, utility_arr):
    
    while(len(utility_arr) < 60):
        utility_arr.insert(0, 0)
    while(len(Q_arr) < 60):
        Q_arr.insert(0, 0)
    while(len(budget_arr) < 60):
        budget_arr.insert(0, 0)


    x_axis = [x for x in range(len(budget_arr))]
    
    fig, ax1 = plt.subplots(dpi = 500)

    
    ax1.plot(x_axis,utility_arr, label='Utility', color='blue', linestyle='dashdot')
    ax1.plot(x_axis, Q_arr, label='Q', color='green', linestyle='dashed')
    ax1.plot(x_axis,budget_arr, label='Cost', color='red' )

    ax1.set_xlabel("Time Frame, T", fontsize=15,  fontweight='bold')
    ax1.set_ylabel('Performance', fontsize=15, fontweight='bold')
    
    ax1.legend(fontsize= 15)
    # plt.show()
    plt.grid()
    file_name=f"combined_fed_{id}.pdf"
    plt.savefig(file_name, dpi = 500)


def graph_combine2(id, budget_arr, L2norm, utility_arr):
    
    while(len(utility_arr) < 60):
        utility_arr.insert(0, 0)
    while(len(L2norm) < 60):
        L2norm.insert(0, 0)
    while(len(budget_arr) < 60):
        budget_arr.insert(0, 0)


    x_axis = [x for x in range(len(budget_arr))]
    
    fig, ax1 = plt.subplots(dpi = 500)

    
    ax1.plot(x_axis,utility_arr, label='Utility', color='blue', linestyle='dashdot')
    ax1.plot(x_axis,L2norm, label='L2Norm-i', color='green', linestyle='dashed')
    ax1.plot(x_axis,budget_arr, label='Budget', color='red' )

    ax1.set_xlabel("Communication round, R", fontsize=15,  fontweight='bold')
    
    
    ax1.legend(fontsize= 15)
    # plt.show()
    plt.grid()
    file_name=f"combined_fed_{id}.pdf"
    plt.savefig(file_name, dpi = 500)



