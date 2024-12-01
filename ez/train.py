# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
import ray
import wandb
import hydra
import torch
import multiprocessing
import glob
# import tqdm
from tqdm.auto import tqdm

import sys
sys.path.append(os.getcwd())

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image

from ez import agents
from ez.utils.format import set_seed, init_logger
from ez.worker import start_workers, join_workers
from ez.eval import eval
from ez.data.trajectory import GameTrajectory
from ez.data.replay_buffer import ReplayBuffer


@hydra.main(config_path='./config', config_name='config', version_base='1.1')
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    if config.ray.single_process:
        config.train.self_play_update_interval = 1
        config.train.reanalyze_update_interval = 1
        config.actors.data_worker = 1
        config.actors.batch_worker = 1
        config.data.num_envs = 1

    if config.ddp.world_size > 1:
        mp.spawn(start_ddp_trainer, args=(config,), nprocs=config.ddp.world_size)
    else:
        start_ddp_trainer(0, config)


def start_ddp_trainer(rank, config):
    assert rank >= 0
    print(f'start {rank} train worker...')
    agent = agents.names[config.agent_name](config)         # update config
    manager = None
    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(num_gpus=num_gpus, num_cpus=num_cpus, object_store_memory=150 * 1024 * 1024 * 1024 if config.env.image_based else 100 * 1024 * 1024 * 1024)
    set_seed(config.env.base_seed + rank >= 0)              # set seed
    # set log

    if rank == 0:
        # wandb logger
        if config.ddp.training_size == 1:
            wandb_name = config.env.game + '-' + config.wandb.tag
            print(f'wandb_name={wandb_name}')
            logger = wandb.init(
                name=wandb_name,
                project=config.wandb.project,
                # config=config,
            )
        else:
            logger = None
        # file logger
        log_path = os.path.join(config.save_path, 'logs')
        os.makedirs(log_path, exist_ok=True)
        init_logger(log_path)
    else:
        logger = None
        
    if config.train.use_demo:
        expert_buffer = load_expert_buffer(config, "./demonstrations")
        bc_model = agent.init_bc(expert_buffer, config)  # Get model back from BC
        # train with pretrained model
        final_weights = train(rank, agent, manager, logger, config, pretrained_model=bc_model)
        
    else:
        final_weights = train(rank, agent, manager, logger, config)

    # final evaluation
    if rank == 0:
        model = agent.build_model()
        model.set_weights(final_weights)
        save_path = Path(config.save_path) / 'recordings' / 'final'

        scores = eval(agent, model, config.train.eval_n_episode, save_path, config)
        print('final score: ', np.mean(scores))
        
def load_expert_buffer(config, demo_dir):
    """Initialize a replay buffer with expert demonstrations stored as .pt files with frames.
    
    Args:
        config: Training configuration
        demo_dir: Path to expert demonstration directory
        
    Returns:
        ray.actor.ActorHandle: Remote replay buffer initialized with expert data
    """
    # Initialize replay buffer server with same config as training
    expert_buffer = ReplayBuffer.remote(
        batch_size=config.train.batch_size,
        buffer_size=config.data.buffer_size,
        top_transitions=config.data.top_transitions, 
        use_priority=config.priority.use_priority,
        env=config.env.env,
        total_transitions=config.data.total_transitions
    )

    # Find all demonstration files
    demo_files = glob.glob(str(Path(demo_dir) / f"{config.env.game}/*.pt"))
    print(f"Loading {len(demo_files)} expert demonstrations from {demo_dir} / {config.env.game} /*.pt")
    
    expert_trajs = []
    priorities = []
    
    for demo_file in tqdm(demo_files):
        # Load demonstration data
        data = torch.load(demo_file)
        
        # Load frames
        frames_dir = Path(os.path.dirname(demo_file)) / "frames"
        frame_fps = [frames_dir / fn for fn in data["frames"]]
        observations = np.stack([np.array(Image.open(fp)) for fp in frame_fps]).transpose(0, 3, 1, 2)
        
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
        
        # Create game trajectory
        traj = GameTrajectory(
            n_stack=config.env.n_stack,
            discount=config.rl.discount,
            gray_scale=config.env.gray_scale,
            unroll_steps=config.rl.unroll_steps,
            td_steps=config.rl.td_steps,
            td_lambda=config.rl.td_lambda,
            obs_shape=config.env.obs_shape,
            trajectory_size=config.env.max_episode_steps, 
            image_based=config.env.image_based,
            episodic=config.env.episodic,
            GAE_max_steps=config.model.GAE_max_steps
        )

        # Initialize trajectory with first n_stack frames
        traj.init([observations[i] for i in range(config.env.n_stack)])
        
        # Add remaining steps
        for i in range(config.env.n_stack, len(observations)):
            # Add step to trajectory
            action = actions[i-config.env.n_stack]
            reward = rewards[i-config.env.n_stack]
            obs = observations[i]
            
            traj.append(action, obs, reward)
            
            # Add placeholder search results
            # Using discounted return as value estimate
            remaining_rewards = rewards[i-config.env.n_stack:]
            discounts = np.array([config.rl.discount**i for i in range(len(remaining_rewards))])
            value_estimate = np.sum(remaining_rewards * discounts)
            
            # For continuous actions, create Gaussian policy centered at expert action
            if config.env.env in ['DMC', 'Gym']:
                policy = np.zeros(config.env.action_space_size * 2)  # Mean and std
                policy[:config.env.action_space_size] = action  # Mean is expert action
                policy[config.env.action_space_size:] = 0.1  # Small fixed std
            else:
                policy = np.zeros(config.env.action_space_size)
                policy[action] = 1.0  # One-hot for discrete actions
                
            traj.store_search_results(
                pred_value=value_estimate,
                search_value=value_estimate, 
                policy=policy
            )
            
            # Add empty snapshot since we don't have MCTS data
            traj.snapshot_lst.append([])
            
        # Save trajectory data
        traj.save_to_memory()
        expert_trajs.append(traj)

    # Save all trajectories to buffer
    for traj in expert_trajs:
        if config.priority.use_priority:
            # Create priorities just for this trajectory
            traj_priorities = np.ones(len(traj)) * getattr(config.priority, 'expert_priority', 1.0)
            ray.get(expert_buffer.save_trajectory.remote(traj, traj_priorities))
        else:
            ray.get(expert_buffer.save_trajectory.remote(traj, None))
        
    print(f"Loaded {len(expert_trajs)} trajectories into expert buffer")
    transition_count = ray.get(expert_buffer.get_transition_num.remote())
    print(f"Total transitions: {transition_count}")
    
    return expert_buffer


def train(rank, agent, manager, logger, config, pretrained_model=None):
    # launch for the main process
    if rank == 0:
        workers, server_lst = start_workers(agent, manager, config)
    else:
        workers, server_lst = None, None

    # train
    storage_server, replay_buffer_server, watchdog_server, batch_storage = server_lst

    if config.ddp.training_size == 1:
        final_weights, final_model = agent.train(rank, replay_buffer_server, storage_server, batch_storage, logger, pretrained_model=pretrained_model)
    else:
        from ez.agents.base import train_ddp
        time.sleep(1)
        train_workers = [
            train_ddp.remote(
                agent, rank * config.ddp.training_size + rank_i,
                replay_buffer_server, storage_server, batch_storage, logger
            ) for rank_i in range(config.ddp.training_size)
        ]
        time.sleep(1)
        final_weights, final_model = ray.get(train_workers)
        

    epi_scores = eval(agent, final_model, 10, Path(config.save_path) / 'evaluation' / 'final', config,
                           max_steps=27000, use_pb=False, verbose=config.eval.verbose)
    print(f'final_mean_score={epi_scores.mean():.3f}')

    # join process
    if rank == 0:
        print(f'[main process] master worker finished')
        time.sleep(1)
        join_workers(workers, server_lst)

    # return
    dist.destroy_process_group()
    return final_weights


if __name__ == '__main__':
    main()
