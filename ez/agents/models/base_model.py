# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import math
import torch.nn as nn
import numpy as np
from .layer import ResidualBlock, conv3x3, mlp


# Down_sample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Add initial pooling layer
        self.initial_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, downsample=self.conv2, stride=2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Initial pooling
        x = self.initial_pool(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class StateToImageNetwork(nn.Module):
    """Maps proprioceptive state [B, state_dim] to image representation [B, C, H, W]"""
    
    def __init__(self, state_dim, target_channels, target_height, target_width, hidden_dims=[256, 512]):
        super().__init__()
        self.state_dim = state_dim
        self.target_channels = target_channels
        self.target_height = target_height
        self.target_width = target_width
        self.target_size = target_channels * target_height * target_width
        
        # Build MLP layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Final layer to target size
        layers.append(nn.Linear(prev_dim, self.target_size))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stable addition
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, state):
        """
        Args:
            state: [B, state_dim] proprioceptive state
        Returns:
            [B, C, H, W] mapped to image representation shape
        """
        batch_size = state.shape[0]
        
        # Pass through MLP
        flat_output = self.mlp(state)
        
        # Reshape to image format
        image_repr = flat_output.view(batch_size, self.target_channels, 
                                    self.target_height, self.target_width)
        
        return image_repr


class HybridRepresentationNetwork(nn.Module):
    """Processes both image and state, combines them, then applies standard processing"""
    
    def __init__(self, observation_shape, state_dim, num_blocks, num_channels, downsample, n_stack):
        super().__init__()
        self.downsample = downsample
        
        # Calculate input channels for stacked images
        input_channels = observation_shape[0] * n_stack  # Account for stacking: 3 * 2 = 6
        
        # Standard image representation network
        if self.downsample:
            self.downsample_net = DownSample(input_channels, num_channels)  # Use stacked channels
            # Calculate downsampled dimensions
            target_height = observation_shape[1] // 32  # Based on DownSample implementation
            target_width = observation_shape[2] // 32
        else:
            self.conv = conv3x3(input_channels, num_channels)  # Use stacked channels
            self.bn = nn.BatchNorm2d(num_channels)
            target_height = observation_shape[1]
            target_width = observation_shape[2]
            
        # State mapping network - account for stacked states
        stacked_state_dim = state_dim * n_stack  # States are also stacked
        self.state_mapper = StateToImageNetwork(
            state_dim=stacked_state_dim,  # Use stacked state dimension
            target_channels=num_channels,
            target_height=target_height,
            target_width=target_width
        )
        
        # Shared residual blocks
        self.resblocks = nn.ModuleList([
            ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)
        ])
        
        # Fusion layer to combine modalities
        self.fusion_conv = conv3x3(num_channels, num_channels)
        self.fusion_bn = nn.BatchNorm2d(num_channels)
        
    def forward(self, hybrid_obs):
        """
        Args:
            hybrid_obs: dict with 'image' [B, C*n_stack, H, W] and 'state' [B, state_dim*n_stack]
        Returns:
            [B, C, H, W] fused representation
        """
        image = hybrid_obs['image']
        state = hybrid_obs['state']
        
        # Process image
        if self.downsample:
            image_repr = self.downsample_net(image)
        else:
            image_repr = self.conv(image)
            image_repr = self.bn(image_repr)
            image_repr = nn.functional.relu(image_repr)
            
        # Process state
        state_repr = self.state_mapper(state)
        
        # Combine representations (element-wise addition)
        combined = image_repr + state_repr
        
        # Apply fusion layer
        fused = self.fusion_conv(combined)
        fused = self.fusion_bn(fused)
        fused = nn.functional.relu(fused)
        
        # Apply residual blocks
        for block in self.resblocks:
            fused = block(fused)
            
        return fused

# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels, downsample):
        """
        Representation network
        :param observation_shape: tuple or list, shape of observations: [C, W, H]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param downsample: bool, True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = conv3x3(
                observation_shape[0],
                num_channels,
            )
            self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, state_size, is_continuous=False, action_embedding=False, action_embedding_dim=32):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, state_size[0], state_size[1]])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, state, action):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        return state


class ValuePolicyNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size,
                 policy_output_size, init_zero, is_continuous=False, policy_distribution='beta', **kwargs):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )
        self.conv1x1_values = nn.ModuleList([nn.Conv2d(num_channels, reduced_channels, 1) for _ in range(self.v_num)])
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_values = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(self.v_num)])
        self.bn_policy = nn.BatchNorm2d(reduced_channels)
        self.block_output_size_value = flatten_size
        self.block_output_size_policy = flatten_size
        self.fc_values = nn.ModuleList([mlp(self.block_output_size_value, fc_layers, value_output_size,
                            init_zero=False if is_continuous else init_zero) for _ in range(self.v_num)])
        self.fc_policy = mlp(self.block_output_size_policy, fc_layers if not is_continuous else [64],
                             policy_output_size, init_zero=init_zero)

        self.is_continuous = is_continuous
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)

        values = []
        for i in range(self.v_num):
            value = self.conv1x1_values[i](x)
            value = self.bn_values[i](value)
            value = nn.functional.relu(value)
            value = value.reshape(-1, self.block_output_size_value)
            value = self.fc_values[i](value)
            values.append(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)
        policy = policy.reshape(-1, self.block_output_size_policy)
        policy = self.fc_policy(policy)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = (torch.nn.functional.softplus(policy[:, action_space_size:] + self.init_std) + self.min_std)#.clip(0, 5)  # same as Dreamer-v3

        return torch.stack(values), policy

class SupportNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1 = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.fc = mlp(flatten_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x):

        x = self.conv1x1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size)
        x = self.fc(x)
        return x


class SupportLSTMNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, lstm_hidden_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flatten_size, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x, hidden):

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.layer(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layer(x)
