import torch
import numpy as np
import math
import random
from scipy.special import gamma


def get_random_problems(batch_size, problem_size, with_distance_matrices=False):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 10:
        demand_scaler = 20
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 150:
        demand_scaler = 60
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)
    
    if with_distance_matrices:
        min_dist, max_dist = convert_xy_to_distance_matrices(depot_xy, node_xy)
        return depot_xy, node_xy, node_demand, min_dist, max_dist
    else:
        return depot_xy, node_xy, node_demand


def convert_xy_to_distance_matrices(depot_xy, node_xy):
    """
    Convert coordinates to min and max distance matrices
    
    Args:
        depot_xy: depot coordinates, shape (batch_size, 1, 2)
        node_xy: node coordinates, shape (batch_size, problem_size, 2)
    
    Returns:
        min_distance_matrix: minimum distances between all nodes (including depot), shape (batch_size, total_nodes, total_nodes)
        max_distance_matrix: maximum distances between all nodes (including depot), shape (batch_size, total_nodes, total_nodes)
    """
    # uncertainty_values = [0.2, 0.4, 0.8]
    uncertainty_values = [0, 0, 0]
    
    batch_size = depot_xy.shape[0]
    problem_size = node_xy.shape[1]
    total_nodes = problem_size + 1
    
    # Combine depot and nodes
    all_xy = torch.cat([depot_xy, node_xy], dim=1)  # shape: (batch_size, total_nodes, 2)
    
    # Calculate Euclidean distances between all nodes
    # Expand dimensions for broadcasting
    x1 = all_xy.unsqueeze(2)  # shape: (batch_size, total_nodes, 1, 2)
    x2 = all_xy.unsqueeze(1)  # shape: (batch_size, 1, total_nodes, 2)
    
    # Calculate Euclidean distance
    min_distance_matrix = torch.norm(x1 - x2, p=2, dim=3)  # shape: (batch_size, total_nodes, total_nodes)
    
    # Create max distance matrix by randomly selecting uncertainty factor for each element
    # Generate random indices for uncertainty values
    random_indices = torch.randint(0, len(uncertainty_values), size=min_distance_matrix.shape, device=min_distance_matrix.device)
    uncertainty_matrix = torch.tensor(uncertainty_values, device=min_distance_matrix.device)[random_indices]
    
    # Apply uncertainty to min distances
    max_distance_matrix = min_distance_matrix * (1 + uncertainty_matrix)
    
    # Set diagonal elements to 0
    eye = torch.eye(total_nodes, device=min_distance_matrix.device).unsqueeze(0).expand(batch_size, -1, -1)
    min_distance_matrix = min_distance_matrix * (1 - eye)
    max_distance_matrix = max_distance_matrix * (1 - eye)
    
    return min_distance_matrix, max_distance_matrix


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