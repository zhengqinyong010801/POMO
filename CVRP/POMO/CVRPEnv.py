from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

import numpy as np
from scipy.stats import norm
import random

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    min_dist: torch.Tensor = None
    # shape: (batch, problem)
    max_dist: torch.Tensor = None
    sample_dist: torch.Tensor = None
@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None
        self.saved_min_dist = None
        self.saved_max_dist = None
        self.saved_sample_dist = None
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_min_dist = loaded_dict['min_dist']
        self.saved_max_dist = loaded_dict['max_dist']
        self.saved_sample_dist = loaded_dict['sample_dist']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, min_dist, max_dist = get_random_problems(batch_size, self.problem_size, with_distance_matrices=True)
            sample_dist = None
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            min_dist = self.saved_min_dist[self.saved_index:self.saved_index+batch_size]
            max_dist = self.saved_max_dist[self.saved_index:self.saved_index+batch_size]
            sample_dist = self.saved_sample_dist[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)


        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.min_dist = min_dist
        self.reset_state.max_dist = max_dist
        self.reset_state.sample_dist = sample_dist

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        """
        Calculate total travel distance using the maximum distance matrix
        Vector-optimized version for faster computation
        """
        # Get the sequence of visited nodes
        selected_node_list = self.selected_node_list  # shape: (batch, pomo, selected_list_length)
        batch_size, pomo_size, selected_list_length = selected_node_list.shape
        
        # Initialize total distance tensor
        travel_distances = torch.zeros(batch_size, pomo_size, device=selected_node_list.device)
        
        # Use the maximum distance matrix
        max_dist_matrix = self.reset_state.min_dist  # shape: (batch, problem+1, problem+1)
        
        # Create batch indices and pomo indices for gathering
        batch_indices = torch.arange(batch_size, device=selected_node_list.device)[:, None, None].expand(batch_size, pomo_size, selected_list_length-1)
        pomo_indices = torch.arange(pomo_size, device=selected_node_list.device)[None, :, None].expand(batch_size, pomo_size, selected_list_length-1)
        
        # Create path segment indices (current and next nodes)
        current_nodes = selected_node_list[:, :, :-1]  # shape: (batch, pomo, selected_list_length-1)
        next_nodes = selected_node_list[:, :, 1:]     # shape: (batch, pomo, selected_list_length-1)
        
        # Gather distances from the max_dist_matrix using advanced indexing
        # For each (batch, pomo, segment) get the distance from max_dist_matrix
        segment_distances = max_dist_matrix[
            batch_indices, 
            current_nodes, 
            next_nodes
        ]
        
        # Sum all segment distances
        travel_distances = segment_distances.sum(dim=2)
        
        # Add the return-to-depot distances (last node to first node)
        last_nodes = selected_node_list[:, :, -1]  # shape: (batch, pomo)
        first_nodes = selected_node_list[:, :, 0]   # shape: (batch, pomo)
        
        # Create batch and pomo indices for the last segment
        batch_indices_last = batch_indices[:, :, 0]  # shape: (batch, pomo)
        
        # Add distances from last to first node
        return_distances = max_dist_matrix[
            batch_indices_last,
            last_nodes,
            first_nodes
        ]
        
        # Add return distances to total distances
        travel_distances += return_distances
        
        return travel_distances
    
    def _get_test_travel_distance(self):
        """
        Calculate total travel distance using the sample distance matrix
        Vector-optimized version for faster computation
        """
        # Get the sequence of visited nodes
        selected_node_list = self.selected_node_list  # shape: (batch, pomo, selected_list_length)
        batch_size, pomo_size, selected_list_length = selected_node_list.shape
        
        # Initialize total distance tensor
        travel_distances = torch.zeros(batch_size, pomo_size, device=selected_node_list.device)
        
        # Check if we have sample_dist available
        assert hasattr(self.reset_state, 'sample_dist') and self.reset_state.sample_dist is not None, "sample_dist matrix must be provided"
        
        # Use the sample distance matrix
        sample_dist_matrix = self.reset_state.sample_dist  # shape: (batch, problem+1, problem+1)
        
        # Create batch indices and pomo indices for gathering
        batch_indices = torch.arange(batch_size, device=selected_node_list.device)[:, None, None].expand(batch_size, pomo_size, selected_list_length-1)
        pomo_indices = torch.arange(pomo_size, device=selected_node_list.device)[None, :, None].expand(batch_size, pomo_size, selected_list_length-1)
        
        # Create path segment indices (current and next nodes)
        current_nodes = selected_node_list[:, :, :-1]  # shape: (batch, pomo, selected_list_length-1)
        next_nodes = selected_node_list[:, :, 1:]     # shape: (batch, pomo, selected_list_length-1)
        
        # Gather distances from the sample_dist_matrix using advanced indexing
        # For each (batch, pomo, segment) get the distance from sample_dist_matrix
        segment_distances = sample_dist_matrix[
            batch_indices, 
            current_nodes, 
            next_nodes
        ]
        
        # Sum all segment distances
        travel_distances = segment_distances.sum(dim=2)
        
        # Add the return-to-depot distances (last node to first node)
        last_nodes = selected_node_list[:, :, -1]  # shape: (batch, pomo)
        first_nodes = selected_node_list[:, :, 0]   # shape: (batch, pomo)
        
        # Create batch indices for the last segment
        batch_indices_last = batch_indices[:, :, 0]  # shape: (batch, pomo)
        
        # Add distances from last to first node
        return_distances = sample_dist_matrix[
            batch_indices_last,
            last_nodes,
            first_nodes
        ]
        
        # Add return distances to total distances
        travel_distances += return_distances
        
        return travel_distances
    