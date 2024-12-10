# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import os
import time
import ray
import torch
from omegaconf import OmegaConf
import torch.distributed as dist
import torch.multiprocessing as mp

from ez.worker.watchdog_worker import start_watchdog_server
from ez.data.global_storage import GlobalStorage
from ez.data.replay_buffer import ReplayBuffer
from ez.worker.data_worker import start_data_worker
from ez.worker.batch_worker import start_batch_worker, start_batch_worker_cpu, start_batch_worker_gpu
from ez.worker.eval_worker import start_eval_worker
from ez.utils.format import RayQueue, PreQueue
from ez.utils.loader import load_expert_buffer
from ez.agents.base import Agent


def start_workers(agent: Agent, manager, config):
    # ==================================================================================================================
    # start server
    # ==================================================================================================================

    # global storage server
    storage_server = GlobalStorage.remote(agent.build_model(), agent.build_model(), agent.build_model())
    print('[main process] Global storage server has been started from main process.')

    # batch queue
    batch_storage = RayQueue(15, 20)
    print('[main process] Batch storage has been initialized.')

    # replay buffer server
    replay_buffer_server = ReplayBuffer.remote(batch_size=config.train.batch_size,
                                               buffer_size=config.data.buffer_size, 
                                               top_transitions=config.data.top_transitions,
                                               use_priority=config.priority.use_priority,
                                               env=config.env.env,
                                               total_transitions=config.data.total_transitions)
    print('[main process] Replay buffer server has been started from main process.')

    if config.train.use_demo:
        # expert buffer server
        expert_buffer_server = ReplayBuffer.remote(batch_size=config.train.batch_size,
                                                buffer_size=config.data.demo_buffer_size, 
                                                top_transitions=config.data.top_transitions,
                                                use_priority=config.priority.use_priority,
                                                env=config.env.env,
                                                total_transitions=config.data.demo_buffer_size)
        print('[main process] Expert buffer server has been started from main process.')

        load_expert_buffer(config, expert_buffer_server, "./demonstrations")
        pretrained_weights = agent.init_bc(expert_buffer_server, config)  # Get model back from BC

        storage_server.set_weights.remote(pretrained_weights, 'self_play')
        storage_server.set_weights.remote(pretrained_weights, 'reanalyze')
        storage_server.set_weights.remote(pretrained_weights, 'latest')
        print('[main process] Initialized models with BC pretrained weights.')

    else:
        expert_buffer_server = None
        pretrained_weights = None

    # watchdog server
    watchdog_server = start_watchdog_server(manager)
    print('[main process] Watchdog server has been started from main process.')
    

    # ==================================================================================================================
    # start worker
    # ==================================================================================================================

    # data workers
    data_workers = [start_data_worker(rank, agent, replay_buffer_server, storage_server, config)
                    for rank in range(0, config.actors.data_worker)]
    print('[main process] Data workers have all been launched.')

    # batch worker
    batch_workers = [start_batch_worker(rank, agent, replay_buffer_server, storage_server, batch_storage, config, expert_buffer=expert_buffer_server)
                     for rank in range(0, config.actors.batch_worker)]
    print('[main process] Batch workers have all been launched.')

    # eval worker
    eval_worker = [start_eval_worker(agent, replay_buffer_server, storage_server, config)]
        

    if int(torch.__version__[0]) == 2:
        print(f'[main process] torch version is {torch.__version__}, enabled torch_compile.')

    # trainer (in current process)
    worker_lst = [data_workers, batch_workers, eval_worker]
    server_lst = [storage_server, replay_buffer_server, watchdog_server, batch_storage, expert_buffer_server]

    return worker_lst, server_lst, pretrained_weights


def join_workers(worker_lst, server_lst):
    data_workers, batch_workers, eval_worker = worker_lst
    storage_server, replay_buffer_server, watchdog_server, smos_server, expert_buffer = server_lst

    # wait for all workers to finish
    for data_worker in data_workers:
        data_worker.join()
    for batch_worker in batch_workers:
        batch_worker.join()
    eval_worker.join()
    print(f'[main process] All workers have stopped.')

    # stop servers
    storage_server.terminate()
    replay_buffer_server.terminate()
    watchdog_server.terminate()
    smos_server.stop()
    print(f'[main process] All servers have stopped.')

