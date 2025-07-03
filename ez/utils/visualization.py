import os
import torch
import numpy as np
from pathlib import Path
import cv2
import ray
from collections import defaultdict

def compare_initial_trajectories(config, replay_buffer, expert_buffer, save_dir=None, num_trajs=2):
    """
    Compare trajectories from expert demonstrations and initial collected self-play data
    by saving videos from observation images.
    
    Args:
        config: Configuration object containing environment settings
        replay_buffer: Ray actor reference to self-play replay buffer
        expert_buffer: Ray actor reference to expert demonstration buffer 
        save_dir: Path to save videos
        num_trajs: Number of trajectory pairs to compare (default 2)
    
    Returns:
        tuple: (expert_trajs, collected_trajs) containing the selected trajectories
    """
    from pathlib import Path
    import imageio
    import numpy as np
    
    # Create save directory if it doesn't exist
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get trajectories by sampling batch contexts
    expert_batch = ray.get(expert_buffer.prepare_batch_context.remote(
        batch_size=num_trajs,
        alpha=1.0,
        beta=1.0,
        rank=0,
        cnt=0
    ))
    expert_batch, _ = expert_batch
    expert_traj_items, expert_positions, *_ = expert_batch
    
    collected_batch = ray.get(replay_buffer.prepare_batch_context.remote(
        batch_size=num_trajs,
        alpha=1.0,
        beta=1.0, 
        rank=0,
        cnt=0
    ))
    collected_batch, _ = collected_batch
    collected_traj_items, collected_positions, *_ = collected_batch
    
    # Save videos for each trajectory pair
    if save_dir:
        # Process expert trajectories
        obs_lst = expert_traj_items[0]  # First element contains observation lists
        for i, observations in enumerate(obs_lst):
            # Get actual observation data from Ray object reference
            observations = ray.get(observations)
            
            print(f"\nExpert trajectory {i} observations shape: {observations[0].shape}")
            
            frames = []
            for obs in observations:
                # Input is already in HWC format (224, 224, 3)
                frame = obs
                
                # Convert to uint8 if normalized
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
            
            if frames:
                print(f"Final frame shape for video: {frames[0].shape}")
                imageio.mimsave(
                    save_dir / f'expert_trajectory_{i}.mp4',
                    frames,
                    fps=30,
                    macro_block_size=1  # Set to 1 to avoid resizing
                )
                
        # Process collected trajectories
        obs_lst = collected_traj_items[0]  # First element contains observation lists
        for i, observations in enumerate(obs_lst):
            # Get actual observation data from Ray object reference
            observations = ray.get(observations)
            
            print(f"\nCollected trajectory {i} observations shape: {observations[0].shape}")
            
            frames = []
            for obs in observations:
                # Input is already in HWC format (224, 224, 3)
                frame = obs
                
                # Convert to uint8 if normalized
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                frames.append(frame)
            
            if frames:
                print(f"Final frame shape for video: {frames[0].shape}")
                imageio.mimsave(
                    save_dir / f'collected_trajectory_{i}.mp4',
                    frames,
                    fps=30,
                    macro_block_size=1  # Set to 1 to avoid resizing
                )
    
    return expert_traj_items, collected_traj_items