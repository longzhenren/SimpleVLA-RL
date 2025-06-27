
#from verl.workers.rollout.rob_rollout import get_robotwin_task
import contextlib
import os
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence


from verl import DataProto
from verl.utils.torch_functional import get_eos_mask


from transformers import GenerationConfig, AutoProcessor

# from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
import numpy as np
from PIL import Image
import tensorflow as tf
from verl import DataProto
#from libero.libero import benchmark
#from codetiming import Timer
from collections import deque
import random
import yaml

import multiprocessing
import gc
from multiprocessing import Process, Queue
from collections import defaultdict



def get_robotwin_args(task_name):
    # TODO (cjh, fix): Assume config has `head_camera_type` attribute, chosen in [L515, D435], otherwise default to D435
    TASK_DESCRIPTIONS = {
        "block_hammer_beat": "There is a hammer and a block in the middle of the table. If the block is closer to the left robotic arm, it uses the left arm to pick up the hammer and strike the block; otherwise, it does the opposite.",
        "block_handover": "A long block is placed on the left side of the table. The left arm grasps the upper side of the block and then hands it over to the right arm, which places the block on the blue mat on the right side of the table.",
        "blocks_stack_easy": "Red and black cubes are placed randomly on the table. The robotic arm stacks the cubes in order, placing the red cubes first, followed by the black cubes, in the designated target location.",
        "blocks_stack_hard": "Red, green, and blue cubes are placed randomly on the table. The robotic arm stacks the cubes in order, placing the red cubes first, followed by the green and then the blue cubes, in the designated target location.",
        "bottle_adjust": "A bottle is placed horizontally on the table. The bottle's design is random and does not repeat in the training and testing sets. When the bottle's head is facing left, pick up the bottle with the right robot arm so that the bottle's head is facing up; otherwise, do the opposite.",
        "container_place": "Random containers (cups, bowls, etc.) are placed randomly on the table. The robotic arm moves the containers into a fixed plate.",
        "diverse_bottles_pick": "A random bottle is placed on the left and right sides of the table. The bottles' designs are random and do not repeat in the training and testing sets. Both left and right arms are used to lift the two bottles to a designated location.",
        "dual_bottles_pick_easy": "A red bottle is placed randomly on the left side, and a green bottle is placed randomly on the right side of the table. Both bottles are standing upright. The left and right arms are used simultaneously to lift the two bottles to a designated location.",
        "dual_bottles_pick_hard": "A red bottle is placed randomly on the left side, and a green bottle is placed randomly on the right side of the table. The bottles' postures are random. Both left and right arms are used simultaneously to lift the two bottles to a designated location.",
        "dual_shoes_place": "One shoe is placed randomly on the left and right sides of the table. The shoes are the same pair with random designs that do not repeat in the training and testing sets. Both left and right arms are used to pick up the shoes and place them in the blue area, with the shoe heads facing the left side of the table.",
        "empty_cup_place": "An empty cup and a cup mat are placed randomly on the left or right side of the table. The robotic arm places the empty cup on the cup mat.",
        "mug_hanging_easy": "A mug is placed randomly on the left side of the table, and a mug rack is placed on the right side (fixed). The left arm moves the mug to a suitable position in the middle of the table, and then the right arm hangs the handle of the mug on the mug rack.",
        "mug_hanging_hard": "A mug is placed randomly on the left side of the table, and a mug rack is placed randomly on the right side. The left arm moves the mug to a suitable position in the middle of the table, and then the right arm hangs the handle of the mug on the mug rack.",
        "pick_apple_messy": "Apples and four random items are placed randomly on the table. The robotic arm picks up the apple and lifts it.",
        "put_apple_cabinet": "Initially, an apple is placed randomly. The robotic arm uses the left arm to open the cabinet and the right arm to pick up the apple and place them inside.",
        "shoe_place": "Shoes are placed randomly on the table, with random designs that do not repeat in the training and testing sets. The robotic arm moves the shoes to a blue area in the center of the table, with the shoe head facing the left side of the table.",
        "tool_adjust": "A tool is placed horizontally on the table. The tool's design is random and does not repeat in the training and testing sets. When the tool's head is facing left, pick up the tool with the right robot arm so that the tool's head is facing up; otherwise, do the opposite."
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
    task_name="block_hammer_beat"
    trial_id=173
    current_seed=173
    env, args = get_robotwin_task(task_name)
    print(f"Setting up demo with seed {current_seed} for task {task_name}, trial_id {trial_id}")
    env.setup_demo(now_ep_num=trial_id, seed=current_seed, is_test=True, **args)
    print(f"Demo setup with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.")
    env.play_once()
    print(f"Demo play once with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.")
    env.close()
    print(f"Env closed with seed {current_seed} for task {task_name}, trial_id {trial_id} successful.")

    print(env.plan_success and env.check_success())

    env.setup_demo(now_ep_num=trial_id, seed=current_seed, is_test=True, **args)
    obs = env.get_obs()


if __name__ == "__main__":
    # 在这里设置 spawn 方法
    

    #env_worker_robotwin()
    # import os
    # print(os.environ)
    processes = []
    input_queues = []
    output_queues = []
    for idx in range(8):
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