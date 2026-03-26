import sys
import os
import csv
import time
import numpy as np
import isaacgym
from isaacgym import gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry
import torch

# ====================== CSV 配置（修复编码错误） ======================
def init_csv(file_path):
    header = [
        "timestamp(s)", "base_x", "base_y", "base_z",
        "FL_x","FL_y","FL_z","FR_x","FR_y","FR_z",
        "RL_x","RL_y","RL_z","RR_x","RR_y","RR_z",
        "FL_contact","FR_contact","RL_contact","RR_contact"
    ]
    # 🔥 修复这里：newline=''，encoding='utf-8'
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def write_csv(file_path, t, base_pos, foot_pos, contact):
    row = [round(t,4), round(float(base_pos[0]),4), round(float(base_pos[1]),4), round(float(base_pos[2]),4)]
    for i in range(4):
        row.extend([round(float(foot_pos[i,0]),4), round(float(foot_pos[i,1]),4), round(float(foot_pos[i,2]),4)])
    row.extend([int(x) for x in contact])
    # 🔥 同步修复这里
    with open(file_path, "a", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ====================== 主函数（直线走+张量正确） ======================
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, f"foot_data_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    init_csv(csv_path)
    start_time = time.time()

    gym = env.gym
    sim = env.sim
    feet_indices = env.feet_indices

    try:
        for i in range(10 * int(env.max_episode_length)):
            # 强制锁死直线行走
            env.commands.zero_()         
            env.commands[:, 0] = 0.5     
            env.commands[:, 1] = 0.0     
            env.commands[:, 2] = 0.0     

            # 兼容obs不报错
            obs_tensor = obs[0].detach() if isinstance(obs, tuple) else obs.detach()
            actions = policy(obs_tensor)
            obs, _, _, dones, _ = env.step(actions)

            # 张量重塑（你已理解的正确逻辑）
            gym.refresh_rigid_body_state_tensor(sim)
            rigid_body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
            rigid_body_state = rigid_body_state.reshape(1, -1, 13) 
            foot_pos = rigid_body_state[0, feet_indices, :3].cpu().numpy()

            # 读取数据
            base_pos = env.base_pos[0].cpu().numpy()
            foot_contact = env.last_contacts[0].cpu().numpy()

            # 写入CSV
            write_csv(csv_path, time.time()-start_time, base_pos, foot_pos, foot_contact)

            if i % 50 == 0:
                print(f"\r步数: {i:4d} | 时间: {time.time()-start_time:.2f}s", end="")

            if dones[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        print(f"\n✅ 运行成功！文件路径：{csv_path}")

if __name__ == '__main__':
    args = get_args()
    play(args)