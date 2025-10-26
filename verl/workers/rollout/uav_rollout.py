# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
UAV Rollout integrated with Isaac Sim via ROS2 (MAVROS) or Pegasus.
Patterned after RobHFRollout, with minimal differences and 'uav' naming.
"""
import contextlib
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.distributed
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
import logging

from transformers import AutoProcessor, GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from .base import BaseRollout

try:
    import rclpy
    from rclpy.duration import Duration
except Exception:
    rclpy = None

__all__ = ["UavHFRollout"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

# --------------------- Utils ---------------------

def center_crop_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return image.crop((left, top, left + s, top + s)).resize((224, 224))


def normalize_proprio(proprio: np.ndarray, norm_stats: dict) -> np.ndarray:
    # Mimic Rob's ACTION_PROPRIO_NORMALIZATION_TYPE=="quantile" behavior
    if not isinstance(proprio, np.ndarray):
        proprio = np.asarray(proprio)
    high = np.asarray(norm_stats.get("q99", np.ones_like(proprio)))
    low = np.asarray(norm_stats.get("q01", -np.ones_like(proprio)))
    return (2.0 * (proprio - low) / (high - low)) - 1.0


# --------------------- Isaac Sim ROS2 Env Wrapper ---------------------
class UavEnvWrapper:
    """Thread-safe UAV environment wrapper using ROS2/MAVROS node.
    - Initializes `/home/user/SimpleVLA-RL/rospy_isaacsim.py:IsaacSimEnv`.
    - Provides `get_obs`, `get_instruction`, `step`, `close`.
    - Action format: [vx, vy, vz, yaw_rate] in body NED.
    - Observation: dict with `full_image` (RGB 224x224) and `state` = [x, y, z].
    """
    def __init__(self, trial_id: int, trial_seed: int, config):
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.config = config
        self.env = None
        self.active = True
        self.complete = False
        self.finish_step = 0
        self.instruction = getattr(self.config, "uav_instruction", "navigate to target and hover")
        self.target_xyz = np.asarray(getattr(self.config, "uav_target_xyz", [0.0, 0.0, 1.0]), dtype=float)
        self.target_tol = float(getattr(self.config, "uav_target_tol", 0.15))
        self.step_dt = float(getattr(self.config, "step_dt", 0.05))
        self.use_stub_image = bool(getattr(self.config, "use_stub_image", True))

    def initialize(self):
        # Import IsaacSimEnv from file path
        import importlib.util
        spec = importlib.util.spec_from_file_location("isaacsim_ros_env", 
            os.path.expanduser("/home/user/SimpleVLA-RL/rospy_isaacsim.py"))
        isaac_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isaac_module)
        IsaacSimEnv = getattr(isaac_module, "IsaacSimEnv")

        logger.info(f"[trial_id={self.trial_id}] Init IsaacSimEnv; stub_image={self.use_stub_image}, target={self.target_xyz.tolist()}, tol={self.target_tol}")
        if rclpy is not None:
            if not rclpy.is_initialized():
                rclpy.init()
        # Create ROS2 node env and reset
        self.env = IsaacSimEnv()
        self.env.reset()
        logger.info(f"[trial_id={self.trial_id}] Env reset complete; OFFBOARD ready")
        self.active = True
        self.complete = False
        self.finish_step = 0
        # Log initial observation snapshot for debugging
        try:
            init_obs = self.get_obs()
            logger.info(f"[trial_id={self.trial_id}] Init obs: state={list(map(float, init_obs['state']))}")
        except Exception as e:
            logger.warning(f"[trial_id={self.trial_id}] Init obs fetch failed: {e}")

    def _get_stub_image(self) -> np.ndarray:
        # 224x224 black image
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def get_obs(self):
        # PoseStamped at /mavros/local_position/pose
        pos = self.env.current_pose.pose.position
        state = np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=float)
        img = None
        if not self.use_stub_image:
            # If you have ROS image topics bridged, adapt here
            img = None
        if img is None:
            img = self._get_stub_image()
        logger.debug(f"[trial_id={self.trial_id}] Obs: state={state.tolist()}, image={'stub' if self.use_stub_image else ('present' if img is not None else 'none')}")
        # Info-level summary for key obs operations
        logger.info(f"[trial_id={self.trial_id}] Obs received: state={state.tolist()}, img_shape={tuple(img.shape)}")
        return {"full_image": img, "state": state}

    def get_instruction(self):
        instr = self.instruction
        logger.info(f"[trial_id={self.trial_id}] Instruction received: {instr}")
        return instr

    def _check_done(self, state: np.ndarray) -> bool:
        return np.linalg.norm(state - self.target_xyz) <= self.target_tol

    def step(self, action: np.ndarray):
        # action shape: (T, 4) or (4,) single step; send velocities at ~20Hz
        if action.ndim == 1:
            action = np.expand_dims(action, 0)
        logger.info(f"[trial_id={self.trial_id}] Executing {len(action)} action(s)")
        for a in action:
            vx, vy, vz, yaw_rate = [float(x) for x in a.tolist()]
            logger.debug(f"[trial_id={self.trial_id}] Cmd send: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw_rate={yaw_rate:.3f}")
            self.env.pub_velocity(vx, vy, vz, yaw_rate)
            if rclpy is not None:
                rclpy.spin_once(self.env, timeout_sec=0.0)
            time.sleep(self.step_dt)
            self.finish_step += 1
            obs = self.get_obs()
            if self._check_done(obs["state"]) or self.finish_step >= int(getattr(self.env, "max_steps", 800)):
                self.active = False
                self.complete = True
                logger.info(f"[trial_id={self.trial_id}] Done at step={self.finish_step}, complete={self.complete}")
                break
        obs = self.get_obs()
        done = not self.active
        return obs, done

    def close(self):
        try:
            if self.env is not None:
                logger.info(f"[trial_id={self.trial_id}] Closing ROS2 node")
                self.env.destroy_node()
        except Exception:
            pass
        try:
            if rclpy is not None and rclpy.is_initialized():
                # Do not shutdown globally in case multiple envs; keep lightweight
                pass
        except Exception:
            pass


# --------------------- Main Rollout ---------------------
class UavHFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        logger.info(f"UavHFRollout init: vla={self.config.vla}, checkpoint={config.pretrained_checkpoint}")
        self.vla_preprocess()

    def vla_preprocess(self):
        # Align unnorm_key with model stats (similar to Rob)
        if self.config.vla in ["openvla-oft"]:
            # For UAV, allow raw key or stripped key
            key = self.config.unnorm_key
            if key not in self.module.norm_stats and key.startswith("uav_"):
                key = key.removeprefix("uav_")
            assert key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA norm_stats!"
            logger.info(f"vla_preprocess: unnorm_key={self.config.unnorm_key} -> {key}")
            self.config.unnorm_key = key

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        # micro batch for generation
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = getattr(self.config, "val_micro_batch_size", None) or 1
        else:
            micro_batch_size = self.config.get("micro_batch_size", batch_size)
        num_chunks = max(batch_size // micro_batch_size, 1)
        logger.info(f"generate_sequences: batch_size={batch_size}, n_samples={prompts.meta_info.get('n_samples')}, micro_bsz={micro_batch_size}, chunks={num_chunks}")
        outputs = [self._generate_minibatch(p) for p in prompts.chunk(chunks=num_chunks)]
        logger.info(f"generate_sequences: finished {len(outputs)} minibatches")
        return DataProto.concat(outputs)

    def process_input(self, inputs: list, task_descriptions: list):
        logger.info(f"process_input: inputs={len(inputs)}; sample_instr={task_descriptions[:min(3, len(task_descriptions))]}")
        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if self.config.get("use_proprio", True):
            batchdata["proprio"] = []
        for i in range(len(inputs)):
            input_data = inputs[i]
            task_description = task_descriptions[i]
            image = Image.fromarray(input_data["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the UAV take to {task_description.lower()}?\nOut:"
            batch_feature = self.processor(prompt, image)
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            logger.debug(f"tokenize[{i}]: ids_shape={tuple(input_ids.shape)}, attn_shape={tuple(attention_mask.shape)}, img_shape={tuple(pixel_values.shape)}")
            # Append tokenizer special token if needed
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
                logger.debug(f"append_eos[{i}]: new_ids_shape={tuple(input_ids.shape)}, new_attn_shape={tuple(attention_mask.shape)}")
            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)
            if self.config.get("use_proprio", True):
                proprio = input_data["state"]
                proprio_norm_stats = self.module.norm_stats[self.config.unnorm_key]["proprio"]
                proprio = normalize_proprio(proprio, proprio_norm_stats)
                batchdata["proprio"].append(torch.from_numpy(proprio))
                logger.debug(f"proprio[{i}]: {proprio}")
        device = torch.device("cuda")
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(device)
            if self.config.get("use_proprio", True):
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
            logger.debug(f"batch_assembled (OFT): ids={tuple(batchdata['input_ids'].shape)}, attn={tuple(batchdata['attention_mask'].shape)}, img={tuple(batchdata['pixel_values'].shape)}")
            logger.info(f"process_input: assembled_batch (OFT) ids={tuple(batchdata['input_ids'].shape)}, attn={tuple(batchdata['attention_mask'].shape)}, img={tuple(batchdata['pixel_values'].shape)}")
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)
            logger.debug(f"batch_assembled: ids={tuple(batchdata['input_ids'].shape)}, attn={tuple(batchdata['attention_mask'].shape)}, img={tuple(batchdata['pixel_values'].shape)}")
            logger.info(f"process_input: assembled_batch ids={tuple(batchdata['input_ids'].shape)}, attn={tuple(batchdata['attention_mask'].shape)}, img={tuple(batchdata['pixel_values'].shape)}")
        return batchdata

    def _prepare_output_batch(self, vla_history, task_records, batch_size):
        batch = {"responses": [], "input_ids": [], "attention_mask": [], "pixel_values": []}
        if self.config.get("use_proprio", True):
            batch["proprio"] = []
        key_names = list(batch.keys())
        for k in key_names:
            for h in vla_history:
                batch[k].append(h[k])
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=1)
        batch["complete"] = torch.tensor([bool(k["complete"]) for k in task_records], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor([k["finish_step"] for k in task_records], dtype=torch.int64, device=batch['responses'].device)
        logger.info(f"prepare_output: responses={tuple(batch['responses'].shape)}, ids={tuple(batch['input_ids'].shape)}, attn={tuple(batch['attention_mask'].shape)}, img={tuple(batch['pixel_values'].shape)}; vla_history_len={len(vla_history)}")
        output_batch = TensorDict(batch, batch_size=batch_size)
        logger.info("prepare_output: TensorDict created; serializing to DataProto")
        return DataProto(batch=output_batch)

    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        batch_size = prompts.batch.batch_size[0] * n_samples
        is_valid = meta_info.get('n_samples') is None
        max_steps = int(getattr(self.config, "uav_max_steps", 800))

        logger.info(f"rollout_minibatch: envs={batch_size}, max_steps={max_steps}, is_valid={is_valid}")
        # Create env wrappers
        env_wrappers = []
        for _ in range(batch_size):
            wrapper = UavEnvWrapper(trial_id=0, trial_seed=0, config=self.config)
            env_wrappers.append(wrapper)
        logger.info(f"rollout_minibatch: created {len(env_wrappers)} env wrappers")
        # Initialize
        for w in env_wrappers:
            w.initialize()
        # Collect initial obs
        inputs, task_descriptions, task_records = [], [], []
        for w in env_wrappers:
            obs = w.get_obs()
            inputs.append(obs)
            instr = w.get_instruction()
            task_descriptions.append(instr)
            logger.info(f"rollout_minibatch: instruction received -> {instr}")
            task_records.append({"active": w.active, "complete": w.complete, "finish_step": w.finish_step, "task_file_name": "uav_trial_0_seed_0"})
        # Rollout loop
        step = 0
        vla_history = []
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            if not active_indices:
                logger.info("rollout_minibatch: all envs inactive; stopping")
                break
            vla_input = self.process_input(inputs, task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            logger.debug(f"model_step[{step}]: responses={tuple(vla_output['responses'].shape)}, actions={np.shape(actions)}")
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step,
            }
            if vla_output.get("proprio") is not None:
                step_data["proprio"] = vla_output["proprio"]
            vla_history.append(step_data)
            # Execute actions
            new_inputs = inputs.copy()
            for idx in active_indices:
                obs, done = env_wrappers[idx].step(actions[idx])
                if obs is not None:
                    new_inputs[idx] = obs
                task_records[idx]['active'] = env_wrappers[idx].active
                task_records[idx]['complete'] = env_wrappers[idx].complete
                task_records[idx]['finish_step'] = env_wrappers[idx].finish_step
                logger.debug(f"env[{idx}] step={task_records[idx]['finish_step']}, active={task_records[idx]['active']}, complete={task_records[idx]['complete']}")
            inputs = new_inputs
            step += self.config.action_chunks_len
        # Cleanup
        for w in env_wrappers:
            try:
                w.close()
            except Exception:
                pass
        logger.info("rollout_minibatch: envs closed; preparing output batch")
        batch_size_td = prompts.batch.batch_size
        # Switch back to train mode after rollout to match RobHFRollout
        self.module.train()
        return self._prepare_output_batch(vla_history, task_records, batch_size_td)

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        if self.config.vla == "openvla-oft":
            return self._generate_one_step_oft(prompts)
        elif self.config.vla == "openvla":
            return self._generate_one_step_openvla(prompts)
        else:
            raise ValueError(f"Unknown VLA type: {self.config.vla}")

    def _generate_one_step_oft(self, prompts: dict):
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)
        param_ctx = contextlib.nullcontext()
        do_sample = prompts.get('do_sample', self.config.do_sample)
        temperature = prompts.get('temperature', self.config.temperature)
        logger.debug(f"generate_oft: do_sample={do_sample}, temp={temperature}, ids={tuple(idx.shape)}, attn={tuple(attention_mask.shape)}, img={tuple(pixel_values.shape)}")
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                actions, response = self.module.generate_action_verl(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    proprio=proprio,
                    attention_mask=attention_mask,
                    padding_idx=self.processor.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    unnorm_key=self.config.unnorm_key,
                    temperature=temperature,
                )
        assert self.processor.tokenizer.pad_token_id is not None
        idx = verl_F.pad_sequence_to_length(
            idx, max_seq_len=self.config.max_prompt_length, pad_token_id=self.processor.tokenizer.pad_token_id, left_pad=True
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask, max_seq_len=self.config.max_prompt_length, pad_token_id=0, left_pad=True
        )
        logger.debug(f"generate_oft_done: responses={tuple(response.shape)}, actions={np.shape(actions)}")
        batch = {
            'responses': response,
            'input_ids': idx,
            'attention_mask': attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        if proprio is not None:
            batch["proprio"] = proprio
        return batch

    def _generate_one_step_openvla(self, prompts: dict):
        idx = prompts['input_ids']
        attention_mask = prompts['attention_mask']
        pixel_values = prompts["pixel_values"]
        eos_token_id = prompts['eos_token_id']
        pad_token_id = prompts['pad_token_id']
        batch_size = idx.size(0)
        prompt_length = idx.size(1)
        param_ctx = contextlib.nullcontext()
        do_sample = prompts.get('do_sample', self.config.do_sample)
        response_length = self.module.get_action_dim(self.config.unnorm_key)
        top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.get('top_k', self.config.get('top_k', 0)) or 0
        top_k = max(0, top_k)
        temperature = prompts.get('temperature', self.config.temperature)
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)
        logger.debug(f"generate_openvla: do_sample={do_sample}, temp={temperature}, top_p={top_p}, top_k={top_k}, resp_len={response_length}, eos={eos_token_id}, pad={pad_token_id}")
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    output_scores=False,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
        seq = output.sequences
        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]
        response_attention_mask = get_eos_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # Discretized tokens to continuous actions based on model bin_centers
        predicted_action_token_ids = response.detach().cpu().numpy()
        discretized_actions = self.module.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=self.module.bin_centers.shape[0] - 1,
        )
        normalized_actions = self.module.bin_centers[discretized_actions]
        action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"]) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        actions = np.expand_dims(actions, axis=1)
        logger.debug(f"generate_openvla_done: responses={tuple(response.shape)}, actions={np.shape(actions)}")
        prompt = verl_F.pad_sequence_to_length(
            prompt, max_seq_len=self.config.max_prompt_length, pad_token_id=self.processor.tokenizer.pad_token_id, left_pad=True
        )
        seq = verl_F.pad_sequence_to_length(
            seq, max_seq_len=self.config.max_prompt_length, pad_token_id=self.processor.tokenizer.pad_token_id, left_pad=True
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask, max_seq_len=self.config.max_prompt_length, pad_token_id=0, left_pad=True
        )
        batch = {
            'prompts': prompt,
            'responses': response,
            'input_ids': seq,
            'attention_mask': attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }
        return batch

    def _obs_to_input(self, obs, is_robotwin=False, robotwin_version="1.0"):
        """Mirror RobHFRollout interface: convert observation to model input.
        For UAV, env already returns processed dict; perform minimal pass-through.
        """
        try:
            state = np.asarray(obs.get("state")) if isinstance(obs.get("state"), (list, tuple, np.ndarray)) else obs.get("state")
        except Exception:
            state = obs.get("state")
        return {
            "full_image": obs.get("full_image"),
            "state": state,
        }