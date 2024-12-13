# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import torch
import logging
import numpy as np

from pathlib import Path
from torch.cuda.amp import autocast as autocast

from .base import Worker
from ez import mcts
from ez.eval import eval
from ez.utils.loader import concat_trajs
from ez.utils.format import formalize_obs_lst
from ez.data.trajectory import GameTrajectory


# @ray.remote(num_gpus=0.05)
@ray.remote(num_gpus=0.05)
class EvalWorker(Worker):
    def __init__(self, agent, replay_buffer, storage, config, expert_buffer=None):
        super().__init__(0, agent, replay_buffer, storage, config)
        self.expert_buffer = expert_buffer

    def calculate_bc_loss(self, model, batch_size=32):
        """Calculate behavior cloning loss on expert demonstrations."""
        if self.expert_buffer is None:
            return None
            
        model.eval()
        total_bc_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for _ in range(100):  # Sample 100 batches to get stable estimate
                # Sample batch from expert buffer
                batch_context = ray.get(
                    self.expert_buffer.prepare_batch_context.remote(
                        batch_size=batch_size,
                        alpha=self.config.priority.priority_prob_alpha,
                        beta=1.0,
                        rank=0,
                        cnt=n_batches
                    )
                )
                
                # Unpack batch
                batch_context, _ = batch_context
                traj_lst, transition_pos_lst, *_ = batch_context
                [obs_lst, reward_lst, policy_lst, action_lst, *_] = traj_lst
                
                traj_lst = concat_trajs(self.config, traj_lst)
                
                # Process observations and actions
                stacked_obs = []
                target_actions = []
                for b in range(len(obs_lst)):
                    pos = transition_pos_lst[b]
                    traj: GameTrajectory = traj_lst[b]
                    stacked_obs.append(traj.get_index_stacked_obs(pos, extra=-self.config.rl.unroll_steps, padding=True))
                    action = action_lst[b][pos]
                    target_actions.append(action)

                obs_batch = formalize_obs_lst(stacked_obs, self.config.env.image_based)
                action_batch = torch.from_numpy(np.array(target_actions)).float().cuda()
                
                # Forward pass
                with autocast():
                    _, _, policies = model.initial_inference(obs_batch, training=False)
                    
                    # Compute BC loss
                    if self.config.env.env in ['DMC', 'Gym']:
                        bc_loss = F.mse_loss(
                            policies[:, :policies.shape[-1] // 2],  # Mean of policy distribution
                            action_batch
                        )
                    else:
                        bc_loss = F.cross_entropy(policies, action_batch.squeeze(-1))
                
                total_bc_loss += bc_loss.item()
                n_batches += 1
                
        return total_bc_loss / n_batches

    def run(self):
        model = self.agent.build_model()
        if int(torch.__version__[0]) == 2:
            model = torch.compile(model)
        best_eval_score = float('-inf')
        episodes = 0
        counter = 0
        eval_steps = 27000 #if self.config.env.game in ['CrazyClimber', 'UpNDown', 'DemonAttack', 'Asterix', 'KungFuMaster'] else 3000           # due to time limitation, eval 3000 steps (instead of 27000) during training.
        
        while not self.is_finished(counter):
            counter = ray.get(self.storage.get_counter.remote())

            if counter >= self.config.train.eval_interval * episodes:
                print('[Eval] Start evaluation at step {}.'.format(counter))
                episodes += 1
                
                model.set_weights(ray.get(self.storage.get_weights.remote('self_play')))
                model.eval()
                
                save_path = Path(self.config.save_path) / 'evaluation' / 'step_{}'.format(counter)
                save_path.mkdir(parents=True, exist_ok=True)
                model_path = Path(self.config.save_path) / 'model.p'
                
                eval_score = eval(self.agent, model, self.config.train.eval_n_episode, save_path, self.config,
                                max_steps=eval_steps, use_pb=False, verbose=0)
                
                # Calculate BC loss
                bc_loss = self.calculate_bc_loss(model)
                
                mean_score = eval_score.mean()
                std_score = eval_score.std()
                min_score = eval_score.min()
                max_score = eval_score.max()
                
                if mean_score >= best_eval_score:
                    best_eval_score = mean_score
                    self.storage.set_best_score.remote(best_eval_score)
                    torch.save(model.state_dict(), model_path)
                
                self.storage.set_eval_counter.remote(counter)
                
                eval_metrics = {
                    'eval/mean_score': mean_score,
                    'eval/std_score': std_score,
                    'eval/max_score': max_score,
                    'eval/min_score': min_score,
                }
                
                if bc_loss is not None:
                    eval_metrics['eval/bc_loss'] = bc_loss
                    
                self.storage.add_eval_log_scalar.remote(eval_metrics)
                
                time.sleep(10)


# ======================================================================================================================
# eval worker
# ======================================================================================================================
def start_eval_worker(agent, replay_buffer, storage, config):
    # start data worker
    eval_worker = EvalWorker.remote(agent, replay_buffer, storage, config)
    eval_worker.run.remote()
