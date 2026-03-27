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

# ====================== 核心工具函数 ======================
def quat_to_x_dir(quat):
    """
    从四元数计算机器人本体X轴在世界坐标系中的方向向量
    :param quat: 四元数，格式为 [x, y, z, w]（IsaacGym RootState 标准格式）
    :return: 机器人X轴方向向量 [dir_x, dir_y, dir_z]（世界坐标系）
    """
    x, y, z, w = quat
    # 四元数旋转单位向量(1,0,0)（世界X轴）到机器人本体X轴的公式
    dir_x = 1 - 2*y*y - 2*z*z
    dir_y = 2*x*y + 2*w*z
    dir_z = 2*x*z - 2*w*y
    return np.array([dir_x, dir_y, dir_z])

# ====================== CSV 配置（适配新需求） ======================
def init_csv(file_path):
    """初始化CSV文件，表头移除足端Z轴，新增机器人X轴方向列"""
    header = [
        "timestamp(s)", "base_x", "base_y", "base_z",
        "robot_x_dir_x", "robot_x_dir_y", "robot_x_dir_z",  # 新增：机器人X轴世界方向
        "FL_x","FL_y",  # 仅保留前左腿X/Y
        "FR_x","FR_y",  # 仅保留前右腿X/Y
        "RL_x","RL_y",  # 仅保留后左腿X/Y
        "RR_x","RR_y",  # 仅保留后右腿X/Y
        "FL_contact","FR_contact","RL_contact","RR_contact"  # 保留触地判定
    ]
    with open(file_path, "w", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow(header)

def write_csv(file_path, t, base_pos, robot_x_dir, foot_xy, contact):
    """写入CSV行数据，适配新的字段结构"""
    # 基础信息：时间戳 + 基座坐标 + 机器人X轴方向
    row = [
        round(t,4),
        round(float(base_pos[0]),4), round(float(base_pos[1]),4), round(float(base_pos[2]),4),
        round(float(robot_x_dir[0]),4), round(float(robot_x_dir[1]),4), round(float(robot_x_dir[2]),4)
    ]
    # 足端XY坐标（仅X/Y，无Z）
    for i in range(4):
        row.extend([round(float(foot_xy[i,0]),4), round(float(foot_xy[i,1]),4)])
    # 触地判定（保留）
    row.extend([int(x) for x in contact])
    # 写入文件
    with open(file_path, "a", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# ====================== 主函数（核心逻辑调整） ======================
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 环境配置（单环境、平面地形、无噪声/推拽）
    env_cfg.env.num_envs = 1
    env_cfg.terrain.type = "plane"
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    # 初始化环境和策略
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 初始化CSV文件
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, f"foot_data_{timestamp}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    init_csv(csv_path)
    start_time = time.time()

    # 缓存环境核心对象
    gym = env.gym
    sim = env.sim
    feet_indices = env.feet_indices

    try:
        for i in range(10 * int(env.max_episode_length)):
            # 强制直线行走（X轴0.5m/s，Y/Z角速度为0）
            env.commands.zero_()         
            env.commands[:, 0] = 0.5     
            env.commands[:, 1] = 0.0     
            env.commands[:, 2] = 0.0     

            # 策略推理 & 环境步进
            obs_tensor = obs[0].detach() if isinstance(obs, tuple) else obs.detach()
            actions = policy(obs_tensor)
            obs, _, _, dones, _ = env.step(actions)

            # 1. 获取机器人基座四元数，计算X轴世界方向
            base_quat = env.base_quat[0].cpu().numpy()  # 基座四元数 [x,y,z,w]
            robot_x_dir = quat_to_x_dir(base_quat)

            # 2. 获取足端XY坐标（仅X/Y，移除Z轴）
            gym.refresh_rigid_body_state_tensor(sim)
            rigid_body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
            rigid_body_state = rigid_body_state.reshape(1, -1, 13)  # [num_envs, num_bodies, 13]
            foot_xy = rigid_body_state[0, feet_indices, :2].cpu().numpy()  # 仅取X/Y列

            # 3. 基础数据读取
            base_pos = env.base_pos[0].cpu().numpy()  # 基座坐标
            foot_contact = env.last_contacts[0].cpu().numpy()  # 足端触地判定

            # 4. 写入CSV（传入机器人X轴方向、足端XY）
            write_csv(
                csv_path, 
                time.time()-start_time, 
                base_pos, 
                robot_x_dir, 
                foot_xy, 
                foot_contact
            )

            # 打印进度
            if i % 50 == 0:
                print(f"\r步数: {i:4d} | 时间: {time.time()-start_time:.2f}s", end="")

            # 环境重置（如果当前episode结束）
            if dones[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n⚠️ 手动停止运行")
    finally:
        print(f"\n✅ 数据采集完成！文件路径：\n{csv_path}")

if __name__ == '__main__':
    args = get_args()
    play(args)