# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import torch
import numpy as np
import ray

from ez.utils.format import str_to_arr


class GameTrajectory:
    def __init__(self, **kwargs):
        # self.raw_obs_lst = []
        self.obs_lst = []
        self.reward_lst = []
        self.policy_lst = []
        self.action_lst = []
        self.pred_value_lst = []
        self.search_value_lst = []
        self.bootstrapped_value_lst = []
        self.mc_value_lst = []
        self.temp_obs = []
        self.snapshot_lst = []

        self.n_stack = kwargs.get('n_stack')
        self.discount = kwargs.get('discount')
        self.obs_to_string = kwargs.get('obs_to_string')
        self.gray_scale = kwargs.get('gray_scale')
        self.unroll_steps = kwargs.get('unroll_steps')
        self.td_steps = kwargs.get('td_steps')
        self.td_lambda = kwargs.get('td_lambda')
        self.obs_shape = kwargs.get('obs_shape')
        self.max_size = kwargs.get('trajectory_size')
        self.image_based = kwargs.get('image_based')
        self.episodic = kwargs.get('episodic')
        self.GAE_max_steps = kwargs.get('GAE_max_steps')
        
        # Initialize state_lst for hybrid mode
        if self.image_based == 2:
            self.state_lst = []

    def init(self, init_frames):
        """Initialize trajectory with stacked observations"""
        assert len(init_frames) == self.n_stack

        for obs in init_frames:
            if self.image_based == 2:
                # Hybrid mode: obs should be dict with 'image' and 'state'
                if isinstance(obs, dict):
                    self.obs_lst.append(copy.deepcopy(obs['image']))
                    self.state_lst.append(copy.deepcopy(obs['state']))
                else:
                    raise ValueError(f"Expected dict observation for hybrid mode during init, got {type(obs)}")
            else:
                # Standard mode: obs is image or state directly
                self.obs_lst.append(copy.deepcopy(obs))

    def append(self, action, obs, reward):
        assert self.__len__() <= self.max_size

        # append a transition tuple
        self.action_lst.append(action)
        
        if self.image_based == 2:
            # Hybrid mode: obs should be a dict with 'image' and 'state' keys
            if isinstance(obs, dict):
                self.obs_lst.append(obs['image'])
                self.state_lst.append(obs['state'])
            else:
                raise ValueError(f"Expected dict observation for hybrid mode, got {type(obs)}")
        else:
            # Standard mode (image-only or state-only)
            self.obs_lst.append(obs)
            
        self.reward_lst.append(reward)

    def pad_over(self, tail_obs, tail_rewards, tail_pred_values, tail_search_values, tail_policies):
        """To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next history block
        , which is necessary for the bootstrapped values at the end states of this history block.
        Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next history block.
        Parameters
        ----------
        tail_obs: list
            tail o_t from the next trajectory block
        tail_rewards: list
            tail r_t from the next trajectory block
        tail_pred_values: list
            tail r_t from the next trajectory block (predicted by network)
        tail_search_values: list
            tail v_t from the next trajectory block (search by mcts)
        tail_policies: list
            tail pi_t from the next trajectory block
        """
        assert len(tail_obs) <= self.unroll_steps
        assert len(tail_policies) <= self.unroll_steps

        # Handle observations based on mode
        if self.image_based == 2:
            # Hybrid mode: tail_obs should be list of dicts or dict with lists
            if len(tail_obs) > 0:
                if isinstance(tail_obs, dict):
                    # tail_obs is dict with 'image' and 'state' lists
                    for img in tail_obs['image']:
                        self.obs_lst.append(copy.deepcopy(img))
                    for state in tail_obs['state']:
                        self.state_lst.append(copy.deepcopy(state))
                elif isinstance(tail_obs[0], dict):
                    # tail_obs is list of dicts
                    for obs in tail_obs:
                        self.obs_lst.append(copy.deepcopy(obs['image']))
                        self.state_lst.append(copy.deepcopy(obs['state']))
                else:
                    raise ValueError(f"Invalid tail_obs format for hybrid mode: {type(tail_obs)}")
        else:
            # Standard mode: tail_obs is list of observations
            for obs in tail_obs:
                self.obs_lst.append(copy.deepcopy(obs))

        # Handle other components (same as before)
        for reward in tail_rewards:
            self.reward_lst.append(reward)

        for pred_value in tail_pred_values:
            self.pred_value_lst.append(pred_value)

        for search_value in tail_search_values:
            self.search_value_lst.append(search_value)

        for policy in tail_policies:
            self.policy_lst.append(policy)

        # calculate bootstrapped value
        self.bootstrapped_value_lst = self.get_bootstrapped_value(value_type='prediction')

    def save_to_memory(self):
        """
        post processing the data when a history block is full
        """
        # convert to numpy
        self.obs_lst = ray.put(np.array(self.obs_lst))
        self.reward_lst = np.array(self.reward_lst)
        self.policy_lst = np.array(self.policy_lst)
        self.action_lst = np.array(self.action_lst)
        self.pred_value_lst = np.array(self.pred_value_lst)
        self.search_value_lst = np.array(self.search_value_lst)
        self.bootstrapped_value_lst = np.array(self.bootstrapped_value_lst)
        
        # Handle state_lst for hybrid mode
        if self.image_based == 2:
            self.state_lst = np.array(self.state_lst)

    def make_target(self, index):
        assert index < self.__len__()

        target_obs = self.get_index_stacked_obs(index)
        target_reward = self.reward_lst[index:index+self.unroll_steps+1]
        target_pred_value = self.pred_value_lst[index:index+self.unroll_steps+1]
        target_search_value = self.search_value_lst[index:index+self.unroll_steps+1]
        target_bt_value = self.bootstrapped_value_lst[index:index+self.unroll_steps+1]
        target_policy = self.policy_lst[index:index+self.unroll_steps+1]

        assert len(target_reward) == len(target_pred_value) == len(target_search_value) == len(target_policy)
        return np.array(target_obs), np.array(target_reward), np.array(target_pred_value), np.array(target_search_value), np.array(target_bt_value), np.array(target_policy)

    def store_search_results(self, pred_value, search_value, policy, idx: int = None):
        # store the visit count distributions and value of the root node after MCTS
        if idx is None:
            self.pred_value_lst.append(pred_value)
            self.search_value_lst.append(search_value)
            self.policy_lst.append(policy)
        else:
            self.pred_value_lst[idx] = pred_value
            self.search_value_lst[idx] = search_value
            self.policy_lst[idx] = policy

    def get_gae_value(self, value_type='prediction', value_lst_input=None, index=None, collected_transitions=None):
        td_steps = 1
        assert value_type in ['prediction', 'search']
        if value_type == 'prediction':
            value_lst = self.pred_value_lst
        else:
            value_lst = self.search_value_lst

        if value_lst_input is not None:
            value_lst = value_lst_input

        traj_len = self.__len__()
        assert traj_len

        if index is None:
            td_lambda = self.td_lambda
        else:
            delta_lambda = 0.1 * (
                    collected_transitions - index) / 3e4
            td_lambda = self.td_lambda - delta_lambda
            td_lambda = np.clip(td_lambda, 0.65, self.td_lambda)

        lens = self.GAE_max_steps
        gae_values = np.zeros(traj_len)
        for i in range(traj_len):
            delta = np.zeros(lens)
            advantage = np.zeros(lens + 1)
            cnt = copy.deepcopy(lens) - 1
            for idx in reversed(range(i, i + lens)):
                if idx + td_steps < traj_len:
                    bt_value = (self.discount ** td_steps) * value_lst[idx + td_steps]
                    for n in range(td_steps):
                        bt_value += (self.discount ** n) * self.reward_lst[idx + n]
                else:
                    steps = idx + td_steps - len(value_lst) + 1
                    bt_value = 0 if self.episodic else (self.discount ** (td_steps - steps)) * value_lst[-1]
                    for n in range(td_steps - steps):
                        assert idx + n < len(self.reward_lst)
                        bt_value += (self.discount ** n) * self.reward_lst[idx + n]

                try:
                    delta[cnt] = bt_value - value_lst[idx]
                except:
                    delta[cnt] = 0
                advantage[cnt] = delta[cnt] + self.discount * td_lambda * advantage[cnt + 1]
                cnt -= 1

            value_lst_tmp = np.concatenate((np.asarray(value_lst)[i:i + lens], np.zeros(lens-np.asarray(value_lst)[i:i + lens].shape[0])))
            gae_values_tmp = advantage[:lens] + value_lst_tmp
            gae_values[i] = gae_values_tmp[0]

        return gae_values

    def get_bootstrapped_value(self, value_type='prediction', value_lst_input=None, index=None, collected_transitions=None):
        assert value_type in ['prediction', 'search']
        if value_type == 'prediction':
            value_lst = self.pred_value_lst
        else:
            value_lst = self.search_value_lst

        if value_lst_input is not None:
            value_lst = value_lst_input

        bt_values = []
        traj_len = self.__len__()
        assert traj_len

        if index is None:
            td_steps = self.td_steps
        else:
            delta_td = (collected_transitions - index) // 3e4
            td_steps = self.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, self.td_steps).astype(np.int32)

        for idx in range(traj_len):
            if idx + td_steps < len(value_lst):
                bt_value = (self.discount ** td_steps) * value_lst[idx + td_steps]
                for n in range(td_steps):
                    bt_value += (self.discount ** n) * self.reward_lst[idx + n]
            else:
                steps = idx + td_steps - len(value_lst) + 1
                bt_value = 0 if self.episodic else (self.discount ** (td_steps - steps)) * value_lst[-1]
                for n in range(td_steps - steps):
                    assert idx + n < len(self.reward_lst)
                    bt_value += (self.discount ** n) * self.reward_lst[idx + n]

            bt_values.append(bt_value)
        return bt_values

    def get_zero_obs(self, n_stack, channel_first=True):

        if self.image_based:
            if channel_first:
                return [np.ones(self.obs_shape, dtype=np.uint8) for _ in
                        range(n_stack)]
            else:
                return [np.ones((self.obs_shape[1], self.obs_shape[2], self.obs_shape[0]), dtype=np.uint8)
                        for _ in range(n_stack)]
        else:
            return np.array([np.ones(self.obs_shape, dtype=np.float32) for _ in range(n_stack)])

    def get_index_stacked_obs(self, index, padding=False, extra=0):
        """To obtain an observation of correct format: o[t, t + stack frames + extra len]
        This mirrors the existing pattern for images/states but handles hybrid case
        """
        unroll_steps = self.unroll_steps + extra
        
        if self.image_based == 2:
            # Hybrid mode: apply original simple approach to BOTH images and states
            
            # Handle images - exactly like original
            frames = ray.get(self.obs_lst)[index:index + self.n_stack + unroll_steps]
            if padding:
                pad_len = self.n_stack + unroll_steps - len(frames)
                if pad_len > 0:
                    if len(frames) > 0:
                        pad_frames = [frames[-1] for _ in range(pad_len)]
                        frames = frames + pad_frames
                    else:
                        frames = []
            
            # Handle states - exactly like original (treat states like images)
            states = self.state_lst[index:index + self.n_stack + unroll_steps]
            if padding:
                pad_len = self.n_stack + unroll_steps - len(states)
                if pad_len > 0:
                    if len(states) > 0:
                        pad_states = [states[-1] for _ in range(pad_len)]
                        states = states + pad_states
                    else:
                        states = []
            
            # Return dict format for hybrid processing
            return {'image': frames, 'state': states}
            
        else:
            # Standard mode (image-only or state-only) - original approach
            frames = ray.get(self.obs_lst)[index:index + self.n_stack + unroll_steps]
            if padding:
                pad_len = self.n_stack + unroll_steps - len(frames)
                if pad_len > 0:
                    if len(frames) > 0:
                        pad_frames = [frames[-1] for _ in range(pad_len)]
                        frames = frames + pad_frames
                    else:
                        frames = []
                    
            if self.obs_to_string:
                from ez.utils.format import str_to_arr
                frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
                
            return frames

    def get_current_stacked_obs(self):
        """Return the current stacked observation - mirrors existing pattern"""
        index = len(self.reward_lst)
        
        if self.image_based == 2:
            # Hybrid mode: return dict with current stacked images and states
            frames = ray.get(self.obs_lst)[index:index + self.n_stack]
            states = self.state_lst[index:index + self.n_stack]
            
            return {'image': frames, 'state': states}
        else:
            # Standard mode - unchanged
            frames = ray.get(self.obs_lst)[index:index + self.n_stack]
            
            if self.obs_to_string:
                from ez.utils.format import str_to_arr
                frames = [str_to_arr(obs, self.gray_scale) for obs in frames]
                
            return frames

    def set_inf_len(self):
        self.max_size = 100000000

    def is_full(self):
        # history block is full
        return self.__len__() >= self.max_size

    def __len__(self):
        # if self.length is not None:
        #     return self.length
        return len(self.action_lst)