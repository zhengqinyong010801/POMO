import torch
import numpy as np
import math
from scipy.special import gamma


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 10:
        demand_scaler = 20
    elif problem_size == 15:
        demand_scaler = 25
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)
    all_xy = torch.cat([depot_xy, node_xy], dim=1)  # shape: (batch, problem+1, 2)

    uncertainty_coordinates, samples_and_probs = generate_time_uncertainty(all_xy=all_xy)
    # print(uncertainty_list.shape)
    # calculate distance
    x1 = all_xy.unsqueeze(2)  # shape: (batch, problem+1, 1, 2)
    x2 = all_xy.unsqueeze(1)  # shape: (batch, 1, problem+1, 2)
    node_dist = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=-1))  # shape: (batch, problem+1, problem+1)

    return depot_xy, node_xy, node_demand, node_dist, uncertainty_coordinates, samples_and_probs


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def generate_time_uncertainty(all_xy, shape=3.0, num_samples=4, uncertainty_percent=0.1):
    """
    Returns:
        1. shape: (batch, selected_paths_count, 2, 2) for coordinates
        2. shape: (batch, selected_paths_count, num_samples, 2) for samples and probabilities
    """
    batch_size, num_nodes, _ = all_xy.shape
    num_paths = num_nodes * (num_nodes - 1)
    selected_paths_count = int(num_paths * uncertainty_percent)

    all_paths = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                all_paths.append((i, j))
    
    selected_indices = np.random.choice(len(all_paths), size=selected_paths_count, replace=False)
    selected_paths = [all_paths[idx] for idx in selected_indices]
    
    start_nodes = np.array([path[0] for path in selected_paths])
    end_nodes = np.array([path[1] for path in selected_paths])
   
    start_coords = all_xy[:, start_nodes, :]
    end_coords = all_xy[:, end_nodes, :]
    
    uncertainty_coordinates = torch.stack([start_coords, end_coords], dim=2)

    distances = torch.norm(uncertainty_coordinates[:, :, 0, :] - uncertainty_coordinates[:, :, 1, :], dim=-1)
    
    scale = distances.unsqueeze(-1) / (shape - 1)
    
    gamma_samples = torch.zeros((batch_size, selected_paths_count, num_samples), device=all_xy.device)
    for b in range(batch_size):
        for p in range(selected_paths_count):
            gamma_samples[b, p] = torch.tensor(
                np.random.gamma(
                    shape=shape,
                    scale=scale[b, p].item(),
                    size=num_samples
                ),
                device=all_xy.device
            )
    
    gamma_k = gamma(shape)
    x = gamma_samples
    theta = scale
    
    probabilities = (
        (x ** (shape - 1)) * torch.exp(-x / theta) /
        (gamma_k * theta ** shape)
    )
    
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    samples_and_probs = torch.stack([gamma_samples, probabilities], dim=-1)
    
    return uncertainty_coordinates, samples_and_probs