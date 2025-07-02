
#from verl.workers.rollout.rob_rollout import get_robotwin_task
import contextlib
import os
import torch
import torch.distributed
# from tensordict import TensorDict NO
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence


#from verl import DataProto #NO
#from verl.utils.torch_functional import get_eos_mask


from transformers import GenerationConfig, AutoProcessor

# from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
import numpy as np
from PIL import Image
import tensorflow as tf

#from libero.libero import benchmark
from codetiming import Timer
from collections import deque
import random
import yaml

import multiprocessing
import gc
from multiprocessing import Process, Queue
from collections import defaultdict

# from verl.utils.libero_utils import save_rollout_video
# from verl.utils.vla_utils.openvla_oft.constants import (
#     ACTION_DIM,
#     ACTION_PROPRIO_NORMALIZATION_TYPE,
# )

#add
import contextlib
import os
import torch
import torch.distributed
#from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
import sys
import importlib

#from verl import DataProto
#from verl.utils.torch_functional import get_eos_mask
#import verl.utils.torch_functional as verl_F

#from verl.workers.rollout.base import BaseRollout

from transformers import GenerationConfig, AutoProcessor

# from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
#from verl.utils.libero_utils import save_rollout_video
#from verl.utils.vla_utils.openvla_oft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
import numpy as np
from PIL import Image
import tensorflow as tf
#from verl import DataProto
#from libero.libero import benchmark
#from codetiming import Timer
from collections import deque
import random
import yaml

import multiprocessing
import gc
from multiprocessing import Process, Queue
from collections import defaultdict
import traceback


def print_env_info(process_name="Main"):
    """打印进程中与 GPU/CUDA/PCI 相关的环境变量"""
    print(f"\n{'='*60}")
    print(f"Process: {process_name} (PID: {os.getpid()})")
    print(f"{'='*60}")
    
    # 关键的 CUDA/GPU 相关环境变量
    important_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CUDA_LAUNCH_BLOCKING',
        'CUDA_MODULE_LOADING',
        'CUDA_CACHE_PATH',
        'CUDA_CACHE_DISABLE',
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES',
        'GPU_DEVICE_ORDINAL',
        'PYTORCH_CUDA_ALLOC_CONF',
        'TORCH_CUDA_ARCH_LIST',
        'LD_LIBRARY_PATH',
        'PATH',
    ]
    
    print("GPU/CUDA Environment Variables:")
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        if var in ['LD_LIBRARY_PATH', 'PATH'] and value != 'NOT SET':
            # 只显示包含 cuda/nvidia 的路径
            paths = value.split(':')
            cuda_paths = [p for p in paths if 'cuda' in p.lower() or 'nvidia' in p.lower()]
            if cuda_paths:
                print(f"  {var}: {':'.join(cuda_paths[:3])}...")  # 只显示前3个相关路径
            else:
                print(f"  {var}: <no cuda/nvidia paths found>")
        else:
            print(f"  {var}: {value}")
    
    # 检查是否有其他 CUDA 相关的环境变量
    print("\nOther CUDA-related variables:")
    cuda_vars = [k for k in os.environ.keys() if 'CUDA' in k or 'GPU' in k or 'NVIDIA' in k]
    other_vars = set(cuda_vars) - set(important_vars)
    if other_vars:
        for var in sorted(other_vars):
            print(f"  {var}: {os.environ[var]}")
    else:
        print("  None found")
    
    # 检查 torch 是否已导入以及 CUDA 状态
    if 'torch' in globals() or 'torch' in locals():
        import torch
        print(f"\nPyTorch CUDA Status:")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            try:
                print(f"  Device name: {torch.cuda.get_device_name(0)}")
            except:
                pass


def get_robotwin_args(task_name):
    # TODO (cjh, fix): Assume config has `head_camera_type` attribute, chosen in [L515, D435], otherwise default to D435
    TASK_DESCRIPTIONS = {
        "block_hammer_beat": "There is a hammer and a block in the middle of the table. If the block is closer to the left robotic arm, it uses the left arm to pick up the hammer and strike the block; otherwise, it does the opposite.",
        
    }

    # config_path = os.path.join(
    #     os.path.dirname(__file__),
    #     '..', '..', 'utils', 'vla_utils', 'openvla_oft', 'robotwin', 'configs', '_base_task_config.yml'
    # )
    config_path = "/mnt/petrelfs/lihaozhan/Rob/SimpleVLA-RL-robotwin/verl/utils/vla_utils/openvla_oft/robotwin/configs/_base_task_config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # camera_config_path = os.path.join(
    #     os.path.dirname(__file__),
    #     '..', '..', 'utils', 'vla_utils', 'openvla_oft', 'robotwin', 'configs', '_camera_config.yml'
    # )
    camera_config_path = "/mnt/petrelfs/lihaozhan/Rob/SimpleVLA-RL-robotwin/verl/utils/vla_utils/openvla_oft/robotwin/configs/_camera_config.yml"
    with open(camera_config_path, 'r', encoding='utf-8') as f:
        camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args['task_description'] = TASK_DESCRIPTIONS[task_name]
    args['head_camera_type'] = "D435"#config.get('head_camera_type', 'D435')
    args['head_camera_fovy'] = camera_args[args['head_camera_type']]['fovy']
    args['head_camera_w'] = camera_args[args['head_camera_type']]['w']
    args['head_camera_h'] = camera_args[args['head_camera_type']]['h']
    args['wrist_camera_fovy'] = camera_args[args['wrist_camera_type']]['fovy']
    args['wrist_camera_w'] = camera_args[args['wrist_camera_type']]['w']
    args['wrist_camera_h'] = camera_args[args['wrist_camera_type']]['h']
    args['front_camera_fovy'] = camera_args[args['front_camera_type']]['fovy']
    args['front_camera_w'] = camera_args[args['front_camera_type']]['w']
    args['front_camera_h'] = camera_args[args['front_camera_type']]['h']
    
    return args

def get_robotwin_task(task_name="block_hammer_beat"):
    import sys
    import importlib
    sys.path.append('./verl')
    envs_module = importlib.import_module(f'utils.vla_utils.openvla_oft.robotwin.envs.{task_name}')
    args = {}
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
        args = get_robotwin_args(task_name )
    except:
        raise SystemExit("No Task")
    return env_instance, args

def env_worker_robotwin():
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # import os
    # print(os.environ)
    
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 使用你的 GPU ID
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'  # 如果使用 OpenGL
    
    # # 清理可能的 CUDA 状态
    # import gc
    # gc.collect()
    # #$print_env_info("In process")
    print("------hi  IN env_worker_robotwin-----",flush=True)
    task_name="block_hammer_beat"
    trial_id=173
    current_seed=173
    env, args = get_robotwin_task(task_name)
    print(f"Setting up demo with seed {current_seed} for task {task_name}, trial_id {trial_id}",flush=True)
    env.setup_demo(now_ep_num=trial_id, seed=current_seed, is_test=True, **args)
    print(f"Demo setup with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.",flush=True)
    env.play_once()
    print(f"Demo play once with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.",flush=True)
    env.close()
    print(f"Env closed with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.",flush=True)

    print(env.plan_success and env.check_success())

    env.setup_demo(now_ep_num=trial_id, seed=current_seed, is_test=True, **args)
    obs = env.get_obs()

    print("------hi  FINISH-----",flush=True)

if __name__ == "__main__":
    # 在这里设置 spawn 方法
    
    #multiprocessing.set_start_method("spawn", force=True)
    #env_worker_robotwin()
    # import os
    # print(os.environ)
    #print_env_info("Main Process (after torch import)")
    print("------hi start-----",flush=True)
    processes = []
    input_queues = []
    output_queues = []
    for idx in range(1):
        # task_name = task_suite_name[idx].removeprefix("robotwin_")
        # t_id = task_id[idx][0].item()
        # tr_id = trial_id[idx][0].item()
        # tr_seed = trial_seed[idx][0].item()
        # input_q = Queue()
        # output_q = Queue()
        env_worker =  env_worker_robotwin
        p = Process(
            target=env_worker,
            args=()
        )
        p.start()
        processes.append(p)
        # input_queues.append(input_q)
        # output_queues.append(output_q)