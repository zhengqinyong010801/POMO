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
    node_dist: torch.Tensor = None
    uncertainty_coordinates: torch.Tensor = None
    samples_and_probs: torch.Tensor = None

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
        self.saved_node_dist = None
        self.uncertainty_coordinates = None
        self.samples_and_probs = None

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
        # self.saved_node_dist = loaded_dict['node_dist']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, node_dist, uncertainty_coordinates, samples_and_probs = get_random_problems(batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            node_dist = self.saved_node_dist[self.saved_index:self.saved_index+batch_size]
            uncertainty_coordinates = self.uncertainty_list[self.saved_index:self.saved_index+batch_size]
            samples_and_probs = self.samples_and_probs[self.saved_index:self.saved_index+batch_size]
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
        self.reset_state.node_dist = node_dist
        self.reset_state.uncertainty_coordinates = uncertainty_coordinates
        self.reset_state.samples_and_probs = samples_and_probs

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
        # 计算路径移动总距离
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)  # shape: (batch, pomo)
        # travel_distances = travel_distances.unsqueeze(-1)  # shape: (batch, pomo, 1)

        if travel_distances is not None:
            time_uncertainty = self._involve_time_uncertainty_expected(ordered_seq)  # shape: (batch, pomo)
            travel_distance = time_uncertainty + travel_distances  # shape: (batch, pomo)
        else:
            travel_distance = torch.zeros((self.batch_size, self.pomo_size, 1), dtype=torch.float32)  # shape: (batch, pomo, 1)

        return travel_distance
    
    def _involve_time_uncertainty(self, ordered_seq):
        uncertainty_coordinates = self.reset_state.uncertainty_coordinates
        # (batch, selected_paths_count, 2, 2)
        samples_and_probs = self.reset_state.samples_and_probs
        # (batch, selected_paths_count, num_samples, 2)
        batch_size, pomo_size, seq_length, _ = ordered_seq.shape
        
        # Initialize time uncertainty tensor
        time_uncertainty = torch.zeros((batch_size, pomo_size, seq_length-1), dtype=torch.float32, device=ordered_seq.device)
        
        # Get all adjacent pairs in current sequence
        seq_pairs = torch.stack([ordered_seq[:, :, :-1], ordered_seq[:, :, 1:]], dim=3)
        # shape: (batch, pomo, seq_length-1, 2, 2)
        
        # Calculate original distances for all pairs at once
        original_dists = torch.norm(seq_pairs[..., 0, :] - seq_pairs[..., 1, :], dim=-1)
        # shape: (batch, pomo, seq_length-1)
        
        # Reshape seq_pairs for comparison
        seq_pairs_flat = seq_pairs.view(-1, 2, 2)  # shape: (batch*pomo*seq_length-1, 2, 2)
        
        # Prepare uncertainty coordinates for comparison
        uncertainty_coords_expanded = uncertainty_coordinates.unsqueeze(1).unsqueeze(2)
        uncertainty_coords_expanded = uncertainty_coords_expanded.expand(
            batch_size, pomo_size, seq_length-1, -1, -1, -1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count, 2, 2)
        
        # Compare coordinates using broadcasting
        start_matches = torch.all(
            uncertainty_coords_expanded[..., 0, :] == seq_pairs.unsqueeze(3)[..., 0, :], 
            dim=-1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count)
        end_matches = torch.all(
            uncertainty_coords_expanded[..., 1, :] == seq_pairs.unsqueeze(3)[..., 1, :], 
            dim=-1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count)
        
        # Combine matches
        matches = start_matches & end_matches
        # shape: (batch, pomo, seq_length-1, selected_paths_count)
        
        # Where matches exist, sample from the corresponding distributions
        match_indices = matches.nonzero()  # shape: (num_matches, 4)
        
        if len(match_indices) > 0:
            b, p, s, m = match_indices.unbind(-1)
            
            # Get corresponding samples and probabilities
            samples = samples_and_probs[b, m, :, 0]  # shape: (num_matches, num_samples)
            probs = samples_and_probs[b, m, :, 1]    # shape: (num_matches, num_samples)
            
            # Sample for all matches at once
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)  # shape: (num_matches,)
            sampled_values = samples[torch.arange(len(samples)), sampled_indices]  # shape: (num_matches,)
            
            # Calculate and assign differences
            diffs = sampled_values - original_dists[b, p, s]
            time_uncertainty[b, p, s] = diffs
        
        # Sum along sequence dimension
        time_uncertainty = time_uncertainty.sum(dim=2)
        # shape: (batch, pomo)
        
        return time_uncertainty

    def _involve_time_uncertainty_expected(self, ordered_seq):
        uncertainty_coordinates = self.reset_state.uncertainty_coordinates
        # (batch, selected_paths_count, 2, 2)
        samples_and_probs = self.reset_state.samples_and_probs
        # (batch, selected_paths_count, num_samples, 2)
        batch_size, pomo_size, seq_length, _ = ordered_seq.shape
        
        # Initialize time uncertainty tensor
        time_uncertainty = torch.zeros((batch_size, pomo_size, seq_length-1), dtype=torch.float32, device=ordered_seq.device)
        
        # Get all adjacent pairs in current sequence
        seq_pairs = torch.stack([ordered_seq[:, :, :-1], ordered_seq[:, :, 1:]], dim=3)
        # shape: (batch, pomo, seq_length-1, 2, 2)
        
        # Calculate original distances for all pairs at once
        original_dists = torch.norm(seq_pairs[..., 0, :] - seq_pairs[..., 1, :], dim=-1)
        # shape: (batch, pomo, seq_length-1)
        
        # Prepare uncertainty coordinates for comparison
        uncertainty_coords_expanded = uncertainty_coordinates.unsqueeze(1).unsqueeze(2)
        uncertainty_coords_expanded = uncertainty_coords_expanded.expand(
            batch_size, pomo_size, seq_length-1, -1, -1, -1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count, 2, 2)
        
        # Compare coordinates using broadcasting
        start_matches = torch.all(
            uncertainty_coords_expanded[..., 0, :] == seq_pairs.unsqueeze(3)[..., 0, :], 
            dim=-1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count)
        end_matches = torch.all(
            uncertainty_coords_expanded[..., 1, :] == seq_pairs.unsqueeze(3)[..., 1, :], 
            dim=-1
        )  # shape: (batch, pomo, seq_length-1, selected_paths_count)
        
        # Combine matches
        matches = start_matches & end_matches
        # shape: (batch, pomo, seq_length-1, selected_paths_count)
        
        # Where matches exist, calculate expected value
        match_indices = matches.nonzero()  # shape: (num_matches, 4)
        
        if len(match_indices) > 0:
            b, p, s, m = match_indices.unbind(-1)
            
            # Get corresponding samples and probabilities
            samples = samples_and_probs[b, m, :, 0]  # shape: (num_matches, num_samples)
            probs = samples_and_probs[b, m, :, 1]    # shape: (num_matches, num_samples)
            
            # Calculate expected value: sum(probability * value)
            expected_values = torch.sum(samples * probs, dim=1)  # shape: (num_matches,)
            
            # Calculate difference from original distance
            diffs = expected_values - original_dists[b, p, s]
            
            # Assign to time_uncertainty tensor
            time_uncertainty[b, p, s] = diffs
        
        # Sum along sequence dimension
        time_uncertainty = time_uncertainty.sum(dim=2)
        # shape: (batch, pomo)
        
        return time_uncertainty


    
    # def _get_travel_distance(self):
    #     # 计算路径移动总距离
    #     gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
    #     # shape: (batch, pomo, selected_list_length, 2)
    #     all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
    #     # shape: (batch, pomo, problem+1, 2)

    #     ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    #     # shape: (batch, pomo, selected_list_length, 2)

    #     rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    #     segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
    #     # shape: (batch, pomo, selected_list_length)

    #     travel_distances = segment_lengths.sum(2)
    #     # shape: (batch, pomo)
    #     return travel_distances

