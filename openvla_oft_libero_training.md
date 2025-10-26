# SimpleVLA-RL: OpenVLA-OFT 在 LIBERO 上的 PPO 训练说明

本文档总结并细化项目中 OpenVLA-OFT 在 LIBERO 任务上的强化学习（PPO/GRPO）训练流程、关键组件、配置项以及常见改动方法。适用于运行 `examples/run_openvla_oft_rl_libero.sh` 的实验场景。

## 入口与环境

- 入口脚本：`examples/run_openvla_oft_rl_libero.sh`
- 运行模块：`python -u -m verl.trainer.main_ppo`
- 关键环境变量：
  - `NCCL_DEBUG=WARN`, `TOKENIZERS_PARALLELISM=true`, `CUDA_LAUNCH_BLOCKING=1`, `TORCH_USE_CUDA_DSA=1`
  - `ROBOT_PLATFORM=LIBERO`（LIBERO 任务）；WandB 需设置 `WANDB_API_KEY`
- 检查点覆盖：脚本会执行 `examples/overwrite_vla_ckpt_utils.sh`，将 OFT 的 Prismatic `configuration/modeling/processing` 等文件覆盖到目标 SFT 检查点目录，使 HF `AutoModelForVision2Seq` 能以 OFT 路径正确加载。

## 主流程（Hydra → Ray → Workers → 训练）

1. `main_ppo.py` 加载 `ppo_trainer.yaml`（及 Megatron 变体），入口脚本通过命令行覆盖其配置。
2. 初始化 Ray 资源池与角色映射：
   - `Role.ActorRollout` 与 `Role.RefPolicy` → `RobActorRolloutRefWorker`（FSDP）
   - `Role.Critic` → `CriticWorker`（FSDP）
   - `Role.RewardModel`（可选，PRIME 或 normal）
3. 构建奖励函数：`RobRewardManager`（训练/验证使用不同实例）。
4. 创建 `RayTrainer`，初始化各 Worker，进入 `fit()` 训练循环。

## 组件职责与交互

- `RobActorRolloutRefWorker`（FSDP 混合引擎）
  - `_build_model_optimizer`：注册并加载 OpenVLA-OFT 或 OpenVLA 的 Vision2Seq；初始化 tokenizer；可选开启 LoRA 与 gradient checkpointing；Actor 带优化器与 LR scheduler；Rollout 包裹 `RobHFRollout` 与 `sharding_manager`；RefPolicy 用于对数概率计算。
  - `generate_sequences(prompts)`：调用 `RobHFRollout.generate_sequences` 与环境交互产生 `responses/action/complete` 等，并在 Actor 角色下可重算 `old_log_probs`。
  - `update_actor(data)`：调用 `RobDataParallelPPOActor.update_policy` 完成 PPO 更新；记录并返回指标。
  - `compute_ref_log_prob(data)`：参考策略对数概率；`compute_entropy(data)`：熵度量。
  - `save_checkpoint(local_path, hdfs_path)`：LoRA 模式保存 adapter 并合并为完整模型；否则导出 FSDP full state dict。

- `RobHFRollout`
  - LIBERO：`_generate_minibatch_libero` 使用多进程 worker 与环境交互；每步通过 `_generate_one_step_oft/openvla` 得到动作；收集 `active/complete/finish_step`；输出以 `_prepare_output_batch` 组织。
  - 输入构建：`_obs_to_input` 将观测转为模型输入；`num_images_in_input` 控制视觉输入通道；LIBERO 默认不含 `proprio`（本体状态）。

- `RayTrainer.fit`
  - 从 `BufferedDataLoader` 取 batch → 复制 `n_samples` 次形成多响应 → 调用 `generate_sequences` 得到 rollout → 奖励与过滤 → 计算优势 → 更新 Actor（必要时更新 Critic）→ 按频率验证与保存。

## PPO / GRPO 训练细节

- 采样与批处理：
  - `data.n_samples` 控制每个 prompt 的响应数（如 `8`），Trainer 将每个样本复制 `n_samples` 次并汇合 rollout 输出。
  - `actor_rollout_ref.rollout.micro_batch_size` 控制 rollout 时的分块生成；`log_prob_micro_batch_size` 用于旧策略对数概率重算；`ppo_micro_batch_size` 用于 Actor 的反向更新时的微批大小，通常与 GPU 数量有关。

- 奖励与优势：
  - Verifier 奖励：`RobRewardManager.verify(roll_batch)` 计算格式与准确性得分，叠加到总奖励（系数 `verifier.reward_coef`）。
  - KL：若启用参考策略与 KL 惩罚，`kl_ctrl` 会基于当前 KL 值调节 `beta`；入口脚本默认 `kl_coef=0.0`，训练阶段不加惩罚。
  - RM：`reward_model.enable=True`；当 `rm_coef!=0.` 且逻辑实现完整时，RM 奖励会叠加（当前训练路径以 Verifier 为主）。
  - 优势：入口脚本将 `algorithm.adv_estimator=grpo`（Group Relative Policy Optimization），优势由组内相对得分估计替代常规 GAE；对于 `n_samples>1` 的组，使用组内统计形成相对优势，从而减少对 Critic 的依赖。

- 更新：
  - Actor：基于 `ppo_mini_batch_size` 与 `ppo_epochs` 做多轮小批更新；`clip_ratio_high/low` 控制 PPO 剪切；`entropy_coeff` 可设为 0 关闭熵正则。
  - Critic：LIBERO+GRPO 配置通常禁用 Critic（`use_critic=False`）；若改用 GAE，可启用 `critic` 并在 `RayTrainer` 中更新。

## 数据过滤与格式

- 准确率过滤：`data.filter_accuracy=True` 时，使用 `accuracy_lower_bound/upper_bound` 对样本组的平均分过滤。
- 截断过滤：`data.filter_truncated=True` 时，检测响应长度是否达到 `data.max_response_length` 并剔除可能截断的组。
- 格式过滤：`data.filter_format=True` 时，依据 `RobRewardManager.verify` 的格式指标进行过滤（LIBERO 场景下，通常设为 True 有利于稳定训练）。

## 关键配置与入口脚本覆盖

入口脚本建议值（LIBERO）：

- 数据与采样：
  - `data.task_suite_name=libero_10`
  - `data.n_samples=8`
  - `data.filter_accuracy=True`
  - `data.accuracy_lower_bound=0.1`, `data.accuracy_upper_bound=0.9`
  - `data.max_prompt_length=256`, `data.max_response_length=128`

- 模型与 Rollout：
  - `actor_rollout_ref.model.vla=openvla-oft`
  - `actor_rollout_ref.rollout.num_images_in_input=1`
  - `actor_rollout_ref.rollout.use_proprio=False`
  - `actor_rollout_ref.rollout.temperature=1.6`
  - `actor_rollout_ref.rollout.micro_batch_size=1`
  - `actor_rollout_ref.rollout.log_prob_micro_batch_size=32`
  - `actor_rollout_ref.rollout.center_crop=True`

- Actor（PPO）：
  - `actor_rollout_ref.actor.optim.lr=5e-6`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=128`
  - `actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS`
  - `actor_rollout_ref.actor.grad_clip=1`
  - `actor_rollout_ref.actor.clip_ratio_high=0.28`, `clip_ratio_low=0.2`
  - `actor_rollout_ref.actor.entropy_coeff=0.0`

- FSDP 与参考策略：
  - `actor_rollout_ref.actor.fsdp_config.grad_offload=True`, `optimizer_offload=True`
  - `actor_rollout_ref.ref.log_prob_micro_batch_size=32`
  - `actor_rollout_ref.ref.fsdp_config.param_offload=True`

- KL 与训练器：
  - `algorithm.kl_ctrl.kl_coef=0.00`
  - `trainer.save_freq=25`, `trainer.test_freq=4`, `trainer.total_epochs=100`, `trainer.val_before_train=True`

## 常见改动与注意事项

- 切换任务套件：将 `DATASET_NAME` 修改为 `libero_90`、`libero_spatial`、`libero_object`、`libero_goal`；对应地调整 `num_trials_per_task` 与验证频率。
- 打开 KL：将 `algorithm.kl_ctrl.kl_coef` 调整为 `1e-3` 级别，并启用参考策略以稳定训练。
- 启用 LoRA：在 `ppo_trainer.yaml` 设置 `actor_rollout_ref.model.lora_rank>0` 与 `target_modules`；保存时会额外导出 adapter 并自动合并保存完整模型。
- 显存与吞吐：`ppo_micro_batch_size` 与 `log_prob_micro_batch_size`、`rollout.micro_batch_size` 需要结合 GPU 数与显存测定；FSDP 的参数/优化器 offload 能降低峰值显存但会增加 CPU/PCIe 开销。
- 图像输入：`num_images_in_input` 会影响视觉 backbone 的输入通道；OFT 会通过 `vision_backbone.set_num_images_in_input(...)` 与 `norm_stats` 初始化一致性。

## 运行示例

```bash
export WANDB_API_KEY=...  # 必填
export ROBOT_PLATFORM=LIBERO

SFT_MODEL_PATH=/path/to/your/sft_ckpt
CKPT_PATH=/path/to/save/checkpoints

# 覆盖 OFT 组件到 SFT 路径
bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
  data.task_suite_name=libero_10 \
  data.n_samples=8 \
  data.filter_accuracy=True \
  data.accuracy_lower_bound=0.1 \
  data.accuracy_upper_bound=0.9 \
  data.train_batch_size=64 \
  data.val_batch_size=496 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.vla=openvla-oft \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.grad_clip=1 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.rollout.num_images_in_input=1 \
  actor_rollout_ref.rollout.use_proprio=False \
  actor_rollout_ref.rollout.temperature=1.6 \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
  actor_rollout_ref.rollout.center_crop=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  trainer.project_name=SimpleVLA-RL \
  trainer.experiment_name="libero10_openvlaoft_grpo" \
  trainer.default_local_dir=$CKPT_PATH/SimpleVLA-RL/libero10_openvlaoft_grpo \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=25 \
  trainer.test_freq=4 \
  trainer.total_epochs=100 \
  trainer.val_only=False \
  algorithm.adv_estimator=grpo \
  trainer.wandb_mode=online \
  trainer.val_before_train=True
```

---

如需扩展到 Robotwin2.0，请参考 `examples/run_openvla_oft_rl_twin2.sh`，并在 `examples/robotwin2_tasks_info.txt` 中根据任务设置 `traj_mini_batch_size` 与 `use_proprio=True`。

## FAQ：模型驻留、交互、Rollout 与参数更新

- 模型是否常驻加载？状态如何维护？
  - `OpenVLA`/`OpenVLA-OFT` 在每个 `RobActorRolloutRefWorker` 初始化阶段通过 `_build_model_optimizer` 加载一次并注册到 FSDP（可启用 `param_offload` 与 `grad_offload`）。逻辑上“常驻”于该 Worker；物理上参数可能在 GPU/CPU 间按需加载与卸载。
  - 生成与更新前后，Worker 会通过 `sharding_manager`/FSDP 控制权重、优化器状态的载入与释放：
    - 采样阶段：`generate_sequences` 进入前加载必要参数，完成后卸载以控制显存峰值。
    - 更新阶段：`update_actor`/`update_critic` 开始前加载参数与优化器，完成一步或一轮 `ppo_epochs` 后卸载。
  - 参考策略（RefPolicy）通常为同架构的只读策略副本，单独受 FSDP 管理，用于 `compute_ref_log_prob` 与 KL 度量，亦采用参数按需驻留策略。

- 如何与模型交互（图像/状态输入 → 动作输出）？
  - 入口：`RobHFRollout.generate_sequences` 在每一步调用 `_generate_one_step`（路由到 `_generate_one_step_oft` 或 `_generate_one_step_openvla`）。
  - 输入构建：`process_input`/`_obs_to_input` 将环境观测整理为 `input_ids`（文本 prompt）、`pixel_values`（图像张量，支持 `center_crop` 与多腕部视角）、`proprio`（本体状态，Robotwin 任务用），并通过 HF `AutoProcessor` 完成 tokenization 与特征提取。配置项：
    - `rollout.num_images_in_input` 控制视觉通道数量；OFT 的视觉骨干会通过 `set_num_images_in_input(...)` 与 `norm_stats` 对齐。
    - `rollout.use_proprio` 控制是否拼接机器人本体状态（LIBERO 默认关闭）。
  - 动作生成：`modeling_prismatic.generate_action_verl` 接收 `input_ids`、`pixel_values`/`unnorm_key`、`proprio`、`attention_mask` 等，返回 `actions`（结构化低维控制/离散 token）与 `response`（文本/结构化输出）。
  - 旧策略对数概率：混合引擎配置下，`generate_sequences` 完成采样后可能重算 `old_log_probs`（`rollout.log_prob_micro_batch_size` 控制微批），以便后续 PPO 更新使用。

- 参数微调/训练是如何进行的？
  - LoRA 适配：在 `_build_model_optimizer` 中根据 `model.lora_rank` 与 `target_modules` 挂载 LoRA 适配器；PPO 更新时仅更新 LoRA 权重，从而显著降低显存与通信开销。
  - 反向更新：`RobDataParallelPPOActor.update_policy` 以 `ppo_mini_batch_size`×`ppo_epochs` 的循环对 Actor 做梯度下降；支持 `ppo_micro_batch_size`、梯度裁剪与（可选）熵系数。
  - 保存：`save_checkpoint` 在启用 LoRA 时先保存 adapter，再与基础 VLA 模型合并导出完整权重；未启用 LoRA 时直接保存 FSDP full state dict。
  - 梯度检查点：可选 `enable_gradient_checkpointing` 降低前向峰值显存，适用于长序列与大视觉骨干。

- 什么是 Rollout？并行语义是什么？
  - Rollout 是策略与环境交互的一次或一段序列生成过程，输出包含 `responses/actions/complete/finish_step` 以及用于奖励与优势估计的辅助信息。
  - 项目中存在两类并行：
    - 多环境并行：`RobHFRollout` 为 LIBERO 启动多个环境 worker 并行推进轨迹。
    - 单状态多样本并行：Trainer 将同一 `prompt`/观测复制 `data.n_samples` 次进行多响应采样（如 `8`），用于 GRPO 的组相对优势估计；Robotwin 的 `_generate_minibatch_robotwin` 明确使用 `n_samples` 扩展并行样本。
  - 生成端也支持微批：`rollout.micro_batch_size` 在采样阶段分块前向，平衡显存与吞吐。

- 参数更新的时机是什么？
  - `RayTrainer.fit` 的每个 epoch 内部流程为：取 batch → 复制形成 `n_samples` 组 → `generate_sequences` 与环境交互 → 验证器/可选 RM 计算奖励与过滤 → 计算优势（`adv_estimator=grpo/gae/...`）→ `update_actor` 执行 PPO 更新 → 若 `use_critic=True`（`algorithm.adv_estimator=gae`）则 `update_critic`。
  - KL/参考策略：若启用 KL 控制，会在每个更新周期统计当前 KL 并由 `kl_ctrl` 动态调节 `beta`；参考策略对数概率通过 `compute_ref_log_prob` 获取。

- Actor 与 Critic 分别是什么？项目中对应哪里？
  - Actor（策略模型）：将 `prompt+图像(+本体)` 映射为动作/响应的模型；对应 `RobActorRolloutRefWorker` 中的主策略与其优化器，训练逻辑在 `RobDataParallelPPOActor.update_policy`；交互入口为 `RobHFRollout.generate_sequences` 与 `_generate_one_step_*`。
  - Critic（价值模型）：估计状态或序列的价值，用于 GAE 等优势估计；对应 `fsdp_workers.CriticWorker`，其 `_build_critic_model_optimizer` 以 `AutoModelForCausalLM` 加线性价值头构建模型，训练逻辑在 `DataParallelPPOCritic.update_policy`。当 `algorithm.adv_estimator=grpo/rloo/reinforce_plus_plus` 时通常禁用。

## FAQ（分类版）：数据、环境、训练、模型等

**数据与数据加载器**
- 加载器实现与职责
  - 训练器在 `RayTrainer._create_dataloader` 构建数据集与加载器：
    - 数据集：`LIBERO_Dataset` 或 `Robotwin_Dataset`（位于 `verl/utils/dataset/rob_dataset.py`）。它们不直接加载像素/状态数据，而是产出任务元信息（如 `task_suite_name`、`task_id`、`trial_id`、`trial_seed`、`data_source`）。
    - 加载器：使用 `torch.utils.data.DataLoader` 与自定义 `collate_fn` 将字典列表堆叠为张量/对象数组；训练端用 `BufferedDataLoader` 包装，支持 `add_to_buffer`、`get_from_buffer` 等缓冲管理。
- 动态读取环境数据
  - 真实观测由 `RobHFRollout` 在 rollout 时动态从环境获取：
    - LIBERO：通过多进程 `env_worker`（`Process + Queue`）调用 `get_libero_env`，每步 `env.step(...)` 返回 `obs`，包含 `agentview_image` 等视觉帧；并记录 `active/complete/finish_step`。
    - Robotwin：通过 `RobotwinEnvWrapper` 线程安全封装环境，提供 `initialize/get_obs/get_instruction/step/close`；每步返回编码后的观测（`encode_obs`），包含头部相机、腕部相机图像，以及本体状态（`state`）。
- 环境可否独立运行
  - 可以。LIBERO 环境在独立进程中运行；Robotwin 环境在独立线程中运行。你可以直接调用各自的环境类运行交互（不经训练框架），但在训练中需通过 `RobHFRollout` 的封装以确保线程/进程安全、队列通信与元信息对齐。
- 替换为自定义环境的步骤
  - 提供满足以下接口的环境或包装器：
    - 必备方法：`get_obs()`、`get_instruction()`、`step(action)` 或 `take_action(action)`、属性/方法：`eval_success`（或等效完成标志）、`step_lim`（最大步数）。
    - 在 `RobHFRollout` 中：
      - 替换或新增获取环境的方法（参考 `get_robotwin2_task`、`get_libero_env`）。
      - 修改 `_obs_to_input` 将你的 `obs` 映射到统一的输入结构：`{'full_image', 'wrist_image(s)', 'state'}`，匹配 `process_input` 的处理流程。
      - 若需多进程/多线程并行，参考 LIBERO 的 `env_worker`（`Process+Queue`）或 Robotwin 的 `ThreadPoolExecutor` 实现。
    - 数据集：在 `verl/utils/dataset/rob_dataset.py` 添加你的 `Dataset`，产出最小元信息集（任务名、trial/seed 等），并在 `ray_trainer.py` 路由到你的数据集名称。

**输入对齐与输出格式**
- 多模态对齐
  - 文本：`AutoProcessor(prompt, image)` 生成 `input_ids` 与 `attention_mask`；在 OFT 路径下进行左侧 padding 到 `max_prompt_length`，并按照 `padding_mask` 排序对齐，保证 `attention_mask.ne(0) == input_ids.ne(pad_token)`。
  - 图像：主图（`full_image`）与腕部图（`wrist_*`）分别处理，`pixel_values_list.append(...)` 后按维度 1 拼接：得到形状近似 `[batch_size, num_images, 3, 224, 224]`（具体维度由 `AutoProcessor` 与模型设置决定）。`num_images_in_input` 控制通道数，OFT 的视觉骨干通过 `set_num_images_in_input(...)` 保持一致。
  - 本体：Robotwin 下，`state` 按 `norm_stats`（`q01/q99` 或 `min/max`）归一化到 `[-1,1]`，键为 `proprio`，并堆叠为张量。
- 输出数据结构（rollout → DataProto）
  - `_prepare_output_batch` 将每步的 `responses/input_ids/attention_mask/pixel_values` 按时间维堆叠：
    - `responses`: `[B, S, L_resp_step]`（批大小 B、步数 S、每步响应长度）。
    - `input_ids`/`attention_mask`: `[B, S, max_prompt_length]`（左 pad 对齐）。
    - `pixel_values`: `[B, S, num_images, 3, H, W]`（按相机维度拼接）。
    - 额外：`complete: [B]`（布尔），`finish_step: [B]`（int，已执行的动作步数）。Robotwin 下如启用 `use_proprio`，还包含 `proprio: [B, S, D]`。
  - 注意：`action` 用于环境执行，不在最终 `DataProto` 的批内返回（训练更新依赖 `responses`、优势、旧 log prob 等）。

**频率与时序**
- 输入/动作频率要求
  - 没有硬性频率要求；框架以 `action_chunks_len` 控制每次生成/执行的动作 token 数（步长），并以 `max_steps` 控制单条轨迹的最大步数。环境内部的渲染/实时性由环境自身决定（如 Robotwin2 默认 `render_freq=0`）。
  - Rollout 每步：`process_input` → `_generate_one_step_*` → 执行动作（线程/进程）→ 收集新观测 → `step += action_chunks_len`，直到达到 `max_steps` 或 `complete=True`。

**训练与微批（PPO）**
- 微批的定义与公式
  - 在 `DataParallelPPOActor.update_policy` 中：设 `M=ppo_mini_batch_size`，`m=ppo_micro_batch_size`，则 `gradient_accumulation = M / m`（要求整除）。
  - 对每个 mini-batch（大小 M）：将其划分为 `M/m` 个 micro-batch，每个大小 m；每次微步计算损失 `loss_m`，并做缩放：`loss = loss_m / gradient_accumulation`，逐步 `backward()`，在处理完所有 micro-batch 后执行一次 `optimizer.step()`。等效于在显存受限时，对大小 M 的批进行分次反向累积。
  - 数值示例：`M=128`，`m=32`，则 `gradient_accumulation=4`；假设每样本有效响应长度 `L_resp=128`，则每微步处理 `32 × 128` token 响应；累积 4 次后再 `step()`。
  - 动态批长（可选）：若 `use_dynamic_bsz=True`，则按 `ppo_max_token_len_per_gpu`（如 `20480`）与 `ulysses_sequence_parallel_size` 自动重排 `micro_batches`，使每微步总 token 近似不超过上限。

## 代码示例（与上文段落对应）

**数据与数据加载器**
- 代码示例：RayTrainer 与数据集/加载器

  路径 `verl/trainer/ppo/ray_trainer.py:292-334`

  ```python
  def _create_dataloader(self):   # next fix
      from torch.utils.data import DataLoader
      # TODO: we have to make sure the batch size is divisible by the dp size
      from verl.utils.dataset.rob_dataset import LIBERO_Dataset, Robotwin_Dataset, collate_fn
      if "libero" in self.config.data.task_suite_name:
          self.train_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                              num_trials_per_task=self.config.data.num_trials_per_task,
                                              train_val ="train")
          self.val_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                          num_trials_per_task=self.config.data.num_trials_per_task,
                                          train_val ="valid")
      elif "robotwin" in self.config.data.task_suite_name:
          # (cjh) We assume here that data set names are "robotwin_{task_name}" or "robotwin_all"
          self.train_dataset = Robotwin_Dataset(self.config.data.task_suite_name,
                                                num_trials_per_task=self.config.data.num_trials_per_task,train_val ="train")
          self.val_dataset = Robotwin_Dataset(self.config.data.task_suite_name,
                                              num_trials_per_task=self.config.data.num_trials_per_task,train_val ="valid")
      else:
          raise ValueError(f'Unsupported task suite name: {self.config.data.task_suite_name}')

      self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
                                         batch_size=int(self.config.data.train_batch_size*self.config.data.oversample_factor),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn))
      self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                       batch_size=self.config.data.val_batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       collate_fn=collate_fn)

      assert len(self.train_dataloader) >= 1
      assert len(self.val_dataloader) >= 1

      print(f'Size of train dataloader: {len(self.train_dataloader)}')
      print(f'Size of val dataloader: {len(self.val_dataloader)}')

      total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

      OmegaConf.set_struct(self.config, True)
      with open_dict(self.config):
          self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
          self.config.critic.optim.total_training_steps = total_training_steps
  ```

  路径 `verl/utils/dataset/rob_dataset.py:38-67`（`collate_fn`）

  ```python
  def collate_fn(data_list: list[dict]) -> dict:
      tensors = {}
      non_tensors = {}

      for data in data_list:
          for key, val in data.items():
              if isinstance(val, torch.Tensor):
                  if key not in tensors:
                      tensors[key] = []
                  tensors[key].append(val)
              else:
                  if key not in non_tensors:
                      non_tensors[key] = []
                  non_tensors[key].append(val)

      for key, val in tensors.items():
          tensors[key] = torch.stack(val, dim=0)

      for key, val in non_tensors.items():
          non_tensors[key] = np.array(val, dtype=object)

      output = {}
      output.update(tensors)
      output.update(non_tensors)
      return output
  ```

  路径 `verl/utils/dataset/rob_dataset.py:222-255`（`BufferedDataLoader`）

  ```python
  class BufferedDataLoader:
      def __init__(self, dataloader):
          self.dataloader = dataloader
          self.batch_size = dataloader.batch_size
          self.buffer = []
          self.dataloader_iter = None

      def start_new_epoch(self):
          self.dataloader_iter = iter(self.dataloader)

      def get_next_batch(self):
          try:
              return next(self.dataloader_iter)
          except StopIteration:
              raise StopIteration

      def __len__(self):
          return len(self.dataloader)

      def add_to_buffer(self, samples):
          if len(self.buffer) == 0:
              self.buffer = samples
          else:
              self.buffer = DataProto.concat([self.buffer, samples])

      def get_from_buffer(self, count, dp_size):
          if count > self.buffer_size():
              count = (self.buffer_size() // dp_size) * dp_size
          samples = self.buffer.slice(range(0, count))
          self.buffer = self.buffer.slice(range(count, self.buffer_size()))
          return samples

      def buffer_size(self):
          return len(self.buffer)
  ```

**环境并行与封装**
- 代码示例：环境并行与封装

  路径 `verl/workers/rollout/rob_rollout.py:334-419`（LIBERO `env_worker`）

  ```python
  def env_worker(task_name, task_id, trial_id, config, input_queue, output_queue, is_valid, global_steps, max_steps):
      """Worker process for Libero environments"""
      from libero.libero import benchmark
      benchmark_dict = benchmark.get_benchmark_dict()
      task_suite = benchmark_dict[task_name]()
      task = task_suite.get_task(task_id)
      initial_states = task_suite.get_task_init_states(task_id)
      initial_state = initial_states[trial_id]
      env = None
      while True:
          try:
              env, task_description = get_libero_env(task, config.model_family, resolution=256)
              break
          except:
              ...
      env.reset()
      obs = env.set_init_state(initial_state)
      ...
      output_data = {
          'type': 'step',
          'obs': obs,
          'active': active,
          'complete': complete,
          'finish_step': finish_step,
          'valid_images': step_images.copy() if is_valid else []
      }
  ```

  路径 `verl/workers/rollout/rob_rollout.py:294-323`（Robotwin `RobotwinEnvWrapper.step`）

  ```python
  def step(self, action):
      """Execute action in environment"""
      with self.lock:
          try:
              self.env.take_action(action)
              done = self.env.eval_success
          except Exception as e:
              done = False
              ...
          try:
              obs = self.env.get_obs()
              obs = encode_obs(obs)
          except Exception as e:
              ...
              obs = None
          self.finish_step += action.shape[0]
          if done or self.finish_step >= self.env.step_lim:
              self.active = False
              self.complete = done
          return obs, done
  ```

**输入映射与输出批组织**
- 代码示例：输入映射

  路径 `verl/workers/rollout/rob_rollout.py:1085-1126`（`_obs_to_input`）

  ```python
  def _obs_to_input(self, obs, is_robotwin=False, robotwin_version="1.0"):
      """Convert observation to model input format"""
      if not is_robotwin:
          # Libero
          state = np.concatenate([...])
          if self.config.num_images_in_input > 1:
              return {"full_image": ..., "wrist_image": ..., "state": state}
          else:
              return {"full_image": ..., "state": state}
      else:
          # Robotwin
          if robotwin_version == "1.0":
              state = obs['joint_action']
              state[6] /= 0.045
              state[13] /= 0.045
          else:  # 2.0
              state = obs['joint_action']['vector']
          if self.config.num_images_in_input == 3:
              return {"full_image": ..., "left_wrist": ..., "right_wrist": ..., "state": state}
          else:
              return {"full_image": ..., "state": state}
  ```

- 代码示例：输出批组织

  路径 `verl/workers/rollout/rob_rollout.py:888-913`（`_prepare_output_batch`）

  ```python
  def _prepare_output_batch(self, vla_history, task_records, batch_size):
      """Prepare the output batch from VLA history"""
      batch = {'responses': [], 'input_ids': [], 'attention_mask': [], 'pixel_values': []}
      key_names = ["responses", "input_ids", "attention_mask", "pixel_values"]
      if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
          batch["proprio"] = []
          key_names.append("proprio")
      for k in key_names:
          for h in vla_history:
              batch[k].append(h[k])
      for k, v in batch.items():
          batch[k] = torch.stack(v, dim=1)
      batch["complete"] = torch.tensor([bool(k["complete"]) for k in task_records], dtype=torch.bool, device=batch['responses'].device)
      batch["finish_step"] = torch.tensor([k["finish_step"] for k in task_records], dtype=torch.int64, device=batch['responses'].device)
      output_batch = TensorDict(batch, batch_size=batch_size)
      return DataProto(batch=output_batch)
  ```

- 代码示例：VLA 预处理与输入组装

  路径 `verl/workers/rollout/rob_rollout.py:475-489`（`vla_preprocess` 摘要）

  ```python
  def vla_preprocess(self):
      if self.config.vla in ["openvla", "openvla-oft"]:
          gpus = tf.config.experimental.list_physical_devices('GPU')
          ...
      if self.config.vla in ["openvla-oft"]:
          if "libero" in self.config.task_suite_name:
              if self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                  self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
          elif "robotwin" in self.config.task_suite_name:
              self.config.unnorm_key = self.config.unnorm_key.removeprefix("robotwin_").removeprefix("robotwin2_")
          assert self.config.unnorm_key in self.module.norm_stats
  ```

  路径 `verl/workers/rollout/rob_rollout.py:504-560`（`process_input` 摘要）

  ```python
  def process_input(self, inputs: list, task_descriptions: list):
      batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
      if self.config.use_proprio and "robotwin" in self.config.task_suite_name:
          batchdata["proprio"] = []
      for i in range(len(inputs)):
          input_data = inputs[i]
          task_description = task_descriptions[i]
          image = Image.fromarray(input_data["full_image"]).convert("RGB")
          if self.config.center_crop:
              image = center_crop_image(image)
          prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
          batch_feature = self.processor(prompt, image)
          pixel_values_list = [batch_feature["pixel_values"]]
          if "robotwin" in self.config.task_suite_name:
              for key in input_data:
                  if "wrist" in key and isinstance(input_data[key], np.ndarray):
                      wrist_image = Image.fromarray(input_data[key]).convert("RGB")
                      if self.config.center_crop:
                          wrist_image = center_crop_image(wrist_image)
                      wrist_batch_feature = self.processor(prompt, wrist_image)
                      pixel_values_list.append(wrist_batch_feature["pixel_values"])
          else:
              if "wrist_image" in input_data:
                  wrist_image = Image.fromarray(input_data["wrist_image"]).convert("RGB")
                  if self.config.center_crop:
                      wrist_image = center_crop_image(wrist_image)
                  wrist_batch_feature = self.processor(prompt, wrist_image)
                  pixel_values_list.append(wrist_batch_feature["pixel_values"])
          batch_feature["pixel_values"] = torch.cat(pixel_values_list, dim=1)
          input_ids = batch_feature["input_ids"]; attention_mask = batch_feature["attention_mask"]
          if not torch.all(input_ids[:, -1] == 29871):
              input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
              if self.config.vla in ["openvla-oft"]:
                  attention_mask = torch.cat((attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1)
          batchdata["input_ids"].append(input_ids)
          batchdata["attention_mask"].append(attention_mask)
          batchdata["pixel_values"].append(batch_feature["pixel_values"])
  ```

**模型接口与统计**
- 代码示例：统一接口与统计

  路径 `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py:1868-1908`（`generate_action_verl` 摘要）

  ```python
  def generate_action_verl(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      unnorm_key: Optional[str] = None,
      proprio=None,
      action_head=None,
      noisy_action_projector=None,
      use_film: bool = False,
      **kwargs: str,
  ) -> np.ndarray:
      """Predict actions from input sequence ... Returns: (unnormalized_actions, action_hidden_states)"""
      pixel_values = kwargs["pixel_values"]
      attention_mask = kwargs["attention_mask"]
      do_sample = kwargs["do_sample"]
      temperature = kwargs["temperature"]
      # Create fake labels tensor (needed for action mask)
      ...
  ```

  路径 `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py:2026-2039`（`get_action_stats`）

  ```python
  def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
      """Get all the logged statistics for the given dataset."""
      unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
      return self.norm_stats[unnorm_key]["action"]
  ```

  路径 `verl/utils/vla_utils/openvla/modeling_prismatic.py:547-562`（OpenVLA `get_action_stats`）

  ```python
  def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
      """Get all the logged statistics for the given dataset."""
      unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
      return self.norm_stats[unnorm_key]["action"]
  ```

**FSDP Workers 交互**
- 代码示例：FSDP Workers 的构建与序列生成

  路径 `verl/workers/fsdp_workers.py:325-351`（`_build_rollout`）

  ```python
  def _build_rollout(self):
      if self.config.rollout.name == 'hf':
          from verl.workers.rollout import RobHFRollout
          from verl.workers.hybrid_engine import BaseShardingManager
          rollout = RobHFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
          sharding_manager = BaseShardingManager()
      elif self.config.rollout.name == 'vllm':
          raise ValueError
      return rollout, sharding_manager
  ```

  路径 `verl/workers/fsdp_workers.py:483-565`（`generate_sequences` 与 `compute_ref_log_prob` 摘要）

  ```python
  @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
  def generate_sequences(self, prompts):
      prompts = prompts.to('cuda')
      recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)
      assert self._is_rollout
      if self._is_offload_param:
          load_fsdp_param_and_grad(...)
      prompts.batch = prompts.batch.cuda()
      prompts.meta_info.update({'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id})
      with self.sharding_manager:
          prompts = self.sharding_manager.preprocess_data(prompts)
          output = self.rollout.generate_sequences(prompts=prompts)
          output = self.sharding_manager.postprocess_data(output)
          torch.cuda.synchronize()

  @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
  def compute_ref_log_prob(self, data: DataProto):
      assert self._is_ref
      data = data.to('cuda')
      if self._is_offload_param:
          load_fsdp_param_and_grad(module=self.ref_module_fsdp, ...)
      micro_batch_size = self.config.ref.log_prob_micro_batch_size
      data.meta_info['micro_batch_size'] = micro_batch_size
      data.meta_info['temperature'] = self.config.rollout.temperature
      ...
      output = self.ref_policy.compute_log_prob(data=data)
      output = DataProto.from_dict(tensors={'ref_log_prob': output})
      ...
      return output
  ```

**一步生成与模型调用**
- 代码示例：`_generate_one_step_oft`

  路径 `verl/workers/rollout/rob_rollout.py:925-959`（`_generate_one_step_oft` 摘要）

  ```python
  def _generate_one_step_oft(self, prompts: dict):
      idx = prompts['input_ids']
      attention_mask = prompts['attention_mask']
      pixel_values = prompts["pixel_values"]
      proprio = prompts.get("proprio", None)
      do_sample = prompts.get('do_sample', self.config.do_sample)
      temperature = prompts.get('temperature', self.config.temperature)
      if isinstance(self.module, FSDP):
          param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
      else:
          param_ctx = contextlib.nullcontext()
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
      ...
  ```

---

## 代码片段索引

- `verl/trainer/ppo/ray_trainer.py:292-334`（`RayTrainer._create_dataloader`）
- `verl/utils/dataset/rob_dataset.py:38-67`（`collate_fn`）
- `verl/utils/dataset/rob_dataset.py:222-255`（`BufferedDataLoader`）
- `verl/workers/rollout/rob_rollout.py:334-419`（`env_worker`）
- `verl/workers/rollout/rob_rollout.py:294-323`（`RobotwinEnvWrapper.step`）
- `verl/workers/rollout/rob_rollout.py:1085-1126`（`_obs_to_input`）
- `verl/workers/rollout/rob_rollout.py:888-913`（`_prepare_output_batch`）
- `verl/workers/rollout/rob_rollout.py:925-959`（`_generate_one_step_oft` 摘要）
- `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py:1868-1908`（`generate_action_verl` 摘要）
- `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py:2026-2039`（`get_action_stats`）
- `verl/utils/vla_utils/openvla/modeling_prismatic.py:547-562`（`get_action_stats`）
- `verl/workers/fsdp_workers.py:325-351`（`_build_rollout`）
- `verl/workers/fsdp_workers.py:483-565`（`generate_sequences` 与 `compute_ref_log_prob`）
- 区分不同微批概念
  - 生成阶段：`rollout.micro_batch_size` 控制前向生成的分块（采样端）。
  - 旧 log prob 重算：`ref.log_prob_micro_batch_size` 控制参考/旧策略 log prob 计算的分块（评估端）。
  - PPO 更新：`actor.ppo_micro_batch_size` 控制训练反向的分块（优化端）。

**FSDP 与显存优化（通俗解释）**
- FSDP 的作用
  - 将大模型参数按层分片到多 GPU，前向/反向时只在需要的 GPU 上临时聚合必要切片，结束后再释放/还原；显著降低单卡的参数驻留显存。
  - 可选的 `param_offload/grad_offload/optimizer_offload` 将部分参数、梯度、优化器状态迁移到 CPU，进一步降低峰值显存，但增加 PCIe/内存带宽开销。
  - 生成步骤中通过 `FSDP.summon_full_params(..., writeback=False)` 在必要时临时拉取完整权重进行推理，避免长期驻留。
- 采用的数据转换与显存优化方法
  - 图像：`center_crop_image` 使用 TF 将图像裁剪/缩放，统一到 `RGB` 与目标分辨率；`AutoProcessor` 进行归一化与张量化。
  - 精度：推理时启用 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`（BF16），兼顾速度与数值稳定；训练可选 `enable_gradient_checkpointing` 降峰值显存。
  - 序列：左侧 padding 与 `padding_mask` 排序，降低多样长度对大模型的非必要开销；`use_remove_padding`/FlashAttention 路径（若启用）可移除 padding 参与的计算。
  - 结构：LoRA 仅训练低秩适配器，减少通信与显存占用；微批与动态批长约束总 token 数；FSDP Offload 控制参数/优化器驻留位置。

**模型支持与兼容性**
- 已支持的 VLA 模型
  - `openvla-oft` 与 `openvla` 的 Vision-to-Sequence 变体（通过 `_build_model_optimizer` 注册/加载）。
- 扩展到其他模型的要求
  - 框架依赖统一接口：`generate_action_verl(input_ids, pixel_values, proprio, attention_mask, ...) → (actions, response)`，以及 `get_action_stats(unnorm_key)`/`norm_stats` 提供动作/本体的反归一化与对齐信息。
  - 视觉骨干需支持可变 `num_images_in_input` 与归一化参数，`AutoProcessor` 能正确处理图像与文本。
  - 因此：
    - 基于 Flow Matching（扩散/流匹配）这类非自回归生成模型，若无法提供上述自回归式 `generate_action_verl` 与离散/连续动作映射，则不直接兼容；需要实现仿射接口或桥接（如将连续动作映射为离散 token，再回译）。
    - `pi0` 等模型若能实现相同接口并提供必要的处理器与归一化统计，可接入；否则需编写适配层与注册到模型仓库（参考 `verl/models/registry.py`）。

**Workers 的通信与协同**
- 角色与通信
  - 使用 Ray 进程集群：`Role.ActorRollout`、`Role.RefPolicy`、`Role.Critic`、`Role.RewardModel` 映射到对应 Worker 类（默认同机多卡）。
  - 训练器创建 `RayWorkerGroup` 并 `spawn/ init_model`，随后在周期内调用 `generate_sequences → verify/filter → update_actor → (update_critic)`；如启用参考策略与 KL 控制，则调用 `compute_ref_log_prob` 与动态 `kl_ctrl`。
- 环境的构建与生命周期
  - Robotwin：在 `RobHFRollout._generate_minibatch_robotwin` 中基于 `task_id/trial_id/trial_seed` 创建 `RobotwinEnvWrapper` 列表，线程池并行执行每步动作与读取观测，直至完成或步数上限；结束时清理环境。
  - LIBERO：在 `RobHFRollout._generate_minibatch_libero` 中为每个样本创建一个 `Process` 与输入/输出队列，循环发送动作、接收观测与状态标志，轨迹结束后发送终止信号并 `join/terminate` 进程。

——
以上条目已根据“环节/方面”分类，并指向具体实现位置，便于你替换环境、对齐多模态输入、理解微批训练与 FSDP 的作用。如需将同样 FAQ 同步到 Robotwin2 说明或 README 的索引中，我可以继续补充。