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
        
        # Load frames
        frames_dir = Path(os.path.dirname(demo_file)) / "frames"
        frame_fps = [frames_dir / fn for fn in data["frames"]]
        observations = np.stack([np.array(Image.open(fp)) for fp in frame_fps])
        # if config.env.obs_to_string:
        #     observations = np.array([arr_to_str(obs.astype(np.uint8)) for obs in observations])
        # observations = observations.transpose(0, 3, 1, 2)
        
        # Get state, actions, rewards
        state = torch.tensor(np.array(data["states"]), dtype=torch.float32)
        
        # Handle different environments
        if config.env.env == "DMC":
            rewards = np.array(data["rewards"])
        else:
            rewards = (np.array([
                info["success" if "success" in info.keys() else "goal_achieved"]
                for info in data["infos"]
            ], dtype=np.float32) - 1.0)
            
        actions = np.array(data["actions"], dtype=np.float32).clip(-1, 1)
        
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

            # Pad from next trajectory if available
            # if config.model.value_target == 'bootstrapped':
            #     gap_step = config.env.n_stack + config.rl.td_steps
            # else:
            #     extra = max(0, min(int(1 / (1 - config.rl.td_lambda)), config.model.GAE_max_steps) - config.rl.unroll_steps - 1)
            #     gap_step = config.env.n_stack + 1 + extra + 1
                
            # if end_idx < len(actions):

            #     next_start = end_idx
            #     pad_obs = observations[next_start:next_start + config.rl.unroll_steps]
            #     pad_rewards = rewards[next_start:next_start + gap_step - 1] 
            #     pad_values = [value_estimate] * gap_step
            #     pad_policies = [policy] * config.rl.unroll_steps
                
            #     traj.pad_over(pad_obs, pad_rewards, pad_values, pad_values, pad_policies)
            # else:
            #     traj.pad_over([], [], [], [], [])

            #take 2

            if config.model.value_target == 'bootstrapped':
                gap_step = config.env.n_stack + config.rl.td_steps
            else:
                extra = max(0, min(int(1 / (1 - config.rl.td_lambda)), config.model.GAE_max_steps) - config.rl.unroll_steps - 1)
                gap_step = config.env.n_stack + 1 + extra + 1

            beg_index = config.env.n_stack
            end_index = beg_index + config.rl.unroll_steps

            pad_obs = observations[beg_index:end_index]

            pad_policies = policy[0:config.rl.unroll_steps]
            pad_values = [value_estimate] * gap_step
            pad_rewards = rewards[0:gap_step - 1]

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
    obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
    bootstrapped_value_lsts = items
    
    traj_lst = []
    for obs_lst, reward_lst, policy_lst, action_lst, pred_value_lst, search_value_lst, bootstrapped_value_lst in \
        zip(obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, bootstrapped_value_lsts):
        
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
        traj_lst.append(traj)
        
    return traj_lst