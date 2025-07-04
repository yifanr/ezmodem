import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import ray
import torch
import glob
# import tqdm
from tqdm.auto import tqdm

import sys
sys.path.append(os.getcwd())

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from PIL import Image

from ez.data.trajectory import GameTrajectory
from ez.data.replay_buffer import ReplayBuffer
from ez.utils.format import arr_to_str

def load_expert_buffer(config, expert_buffer: ReplayBuffer, demo_dir):
    """Initialize a replay buffer with expert demonstrations stored as .pt files with frames.
    
    Args:
        config: Training configuration
        demo_dir: Path to expert demonstration directory
        
    Returns:
        ray.actor.ActorHandle: Remote replay buffer initialized with expert data
    """
    # Find all demonstration files
    demo_files = glob.glob(str(Path(demo_dir) / f"{config.env.game}/*.pt"))
    print(f"Loading {len(demo_files)} expert demonstrations from {demo_dir} / {config.env.game} /*.pt")
    
    for demo_file in tqdm(demo_files):
        # Load demonstration data
        data = torch.load(demo_file)
        
        # Load frames and states
        if config.env.image_based:
            # Load frames for image-based observations
            frames_dir = Path(os.path.dirname(demo_file)) / "frames"
            frame_fps = [frames_dir / fn for fn in data["frames"]]
            image_observations = np.stack([np.array(Image.open(fp)) for fp in frame_fps])
        else:
            image_observations = None
        
        # Get state, actions, rewards
        state_observations = torch.tensor(np.array(data["states"]), dtype=torch.float32).numpy()
        
        # Handle different environments
        if config.env.env == "DMC":
            rewards = np.array(data["rewards"])
        else:
            rewards = (np.array([
                info["success" if "success" in info.keys() else "goal_achieved"]
                for info in data["infos"]
            ], dtype=np.float32) - 1.0)
            
        actions = np.array(data["actions"], dtype=np.float32).clip(-1, 1)
        
        # Create observations based on mode
        if config.env.image_based == 2:
            # Hybrid mode: create dict observations
            observations = []
            for i in range(len(state_observations)):
                obs = {
                    'image': image_observations[i],
                    'state': state_observations[i]
                }
                observations.append(obs)
        elif config.env.image_based == 1:
            # Image-only mode
            observations = image_observations
        else:
            # State-only mode
            observations = state_observations
        
        # Split into multiple trajectories if needed
        for start_idx in range(0, len(actions), config.data.trajectory_size):
            end_idx = min(start_idx + config.data.trajectory_size, len(actions))
            
            traj = GameTrajectory(
                n_stack=config.env.n_stack,
                discount=config.rl.discount,
                gray_scale=config.env.gray_scale, 
                unroll_steps=config.rl.unroll_steps,
                td_steps=config.rl.td_steps,
                td_lambda=config.rl.td_lambda,
                obs_shape=config.env.obs_shape,
                trajectory_size=config.data.trajectory_size,
                image_based=config.env.image_based,
                episodic=config.env.episodic,
                GAE_max_steps=config.model.GAE_max_steps
            )

            # Add steps to trajectory
            for i in range(start_idx, end_idx):
                traj.append(actions[i], observations[i], rewards[i])
                
                # Calculate value estimate and policy
                remaining_rewards = rewards[i:]
                discounts = np.array([config.rl.discount**i for i in range(len(remaining_rewards))])
                value_estimate = np.sum(remaining_rewards * discounts)
                
                policy = np.zeros(config.mcts.num_top_actions)
                policy[0] = 0.9
                policy[1:] = 0.1 / (config.mcts.num_top_actions - 1)
                
                traj.store_search_results(value_estimate, value_estimate, policy)
                traj.snapshot_lst.append([])

            if config.model.value_target == 'bootstrapped':
                extra = config.rl.td_steps
            else:
                # For GAE, need enough steps for lambda-returns
                extra = max(0, min(int(1 / (1 - config.rl.td_lambda)), 
                                 config.model.GAE_max_steps) - config.rl.unroll_steps - 1)

            # Total padding needed for bootstrapping
            pad_length = config.env.n_stack + extra + 1

            # Get padding data from next trajectory if available
            if end_idx + pad_length <= len(observations):
                # Create appropriate padding observations
                if config.env.image_based == 2:
                    # Hybrid mode padding
                    pad_obs = []
                    for idx in range(end_idx, end_idx + config.rl.unroll_steps):
                        pad_obs.append({
                            'image': image_observations[idx],
                            'state': state_observations[idx]
                        })
                else:
                    # Regular padding
                    pad_obs = observations[end_idx:end_idx + config.rl.unroll_steps]
                    
                pad_rewards = rewards[end_idx:end_idx + pad_length - 1]
                
                # Calculate value estimates for padding region
                remaining_rewards = rewards[end_idx:]
                discounts = np.array([config.rl.discount**i for i in range(len(remaining_rewards))])
                pad_values = np.ones(pad_length) * np.sum(remaining_rewards * discounts)
                
                # Use same policy distribution for padding
                pad_policies = []
                for _ in range(config.rl.unroll_steps):
                    policy = np.zeros(config.mcts.num_top_actions)
                    policy[0] = 0.9
                    policy[1:] = 0.1 / (config.mcts.num_top_actions - 1)
                    pad_policies.append(policy)

            else:
                # If we don't have enough future data, pad with zeros/last frame
                if config.env.image_based == 2:
                    # Hybrid mode padding
                    last_obs = {
                        'image': image_observations[end_idx - 1],
                        'state': state_observations[end_idx - 1]
                    }
                    pad_obs = [last_obs] * config.rl.unroll_steps
                else:
                    # Regular padding
                    last_obs = observations[end_idx - 1]
                    pad_obs = np.array([last_obs] * config.rl.unroll_steps)
                    
                pad_rewards = np.zeros(pad_length - 1)
                pad_values = np.zeros(pad_length)
                
                pad_policies = []
                for _ in range(config.rl.unroll_steps):
                    policy = np.zeros(config.mcts.num_top_actions)
                    policy[0] = 0.9
                    policy[1:] = 0.1 / (config.mcts.num_top_actions - 1)
                    pad_policies.append(policy)

            # Apply padding to trajectory
            traj.pad_over(pad_obs, pad_rewards, pad_values, pad_values, pad_policies)
            
            traj.save_to_memory()
            
            # Save trajectory to buffer 
            if config.priority.use_priority:
                priorities = np.ones(len(traj)) * getattr(config.priority, 'expert_priority', 1.0)
                ray.get(expert_buffer.save_trajectory.remote(traj, priorities))
            else:
                ray.get(expert_buffer.save_trajectory.remote(traj, None))

    transition_count = ray.get(expert_buffer.get_transition_num.remote())
    print(f"Loaded {transition_count} expert transitions")
    return expert_buffer

def concat_trajs(config, items):
    # Handle both old format (7 items) and new format (8 items with state_lsts)
    if len(items) == 8:
        obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts, state_lsts = items
    else:
        obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts = items
        state_lsts = None
    
    traj_lst = []
    for i, (obs_lst, reward_lst, policy_lst, action_lst, pred_value_lst, search_value_lst, bootstrapped_value_lst) in \
        enumerate(zip(obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts)):
        
        traj = GameTrajectory(
            n_stack=config.env.n_stack,
            discount=config.rl.discount,
            gray_scale=config.env.gray_scale,
            unroll_steps=config.rl.unroll_steps,
            td_steps=config.rl.td_steps,
            td_lambda=config.rl.td_lambda,
            obs_shape=config.env.obs_shape,
            max_size=config.data.trajectory_size,
            image_based=config.env.image_based,
            episodic=config.env.episodic,
            GAE_max_steps=config.model.GAE_max_steps
        )
        
        traj.obs_lst = obs_lst
        traj.reward_lst = reward_lst
        traj.policy_lst = policy_lst
        traj.action_lst = action_lst
        traj.pred_value_lst = pred_value_lst
        traj.search_value_lst = search_value_lst
        traj.bootstrapped_value_lst = bootstrapped_value_lst
        
        # Handle state_lst for hybrid mode
        if state_lsts is not None and i < len(state_lsts) and state_lsts[i] is not None:
            traj.state_lst = state_lsts[i]
        elif config.env.image_based == 2:
            # Initialize empty state_lst for hybrid mode if not provided
            traj.state_lst = []
                
        traj_lst.append(traj)
        
    return traj_lst