import sys
import os
import csv
import time
import numpy as np
import subprocess

import isaacgym
from isaacgym import gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry

import torch

# ====================== GPU 内存检查 ======================
def check_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        free_memory = int(result.stdout.strip().split('\n')[0])
        print(f"[GPU] 当前可用显存: {free_memory} MB")
        if free_memory < 1024:
            print("⚠️ 警告: 显存可能不足!")
        return free_memory < 1024
    except:
        print("[GPU] 无法检测显存状态")
        return False

# ====================== 工具函数 ======================
def quat_to_x_dir(quat):
    """四元数转机器人正前方(X轴)方向向量"""
    x, y, z, w = quat
    dir_x = 1 - 2*y*y - 2*z*z
    dir_y = 2*x*y + 2*w*z
    dir_z = 2*x*z - 2*w*y
    return np.array([dir_x, dir_y, dir_z])

# ====================== 数据处理模块 (优化后) ======================
# 优化点：使用内存暂存数据，循环结束后一次性写入硬盘，避免拖慢仿真速度
class DataLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_buffer = []
        self.header = [
            "timestamp(s)", "base_x", "base_y", "base_z",
            "robot_x_dir_x", "robot_x_dir_y", "robot_x_dir_z",
            "FL_x","FL_y","FR_x","FR_y","RL_x","RL_y","RR_x","RR_y",
            "FL_contact","FR_contact","RL_contact","RR_contact"
        ]

    def add_record(self, t, base_pos, robot_x_dir, foot_xy, contact):
        row = [
            round(t, 4),
            round(float(base_pos[0]), 4), round(float(base_pos[1]), 4), round(float(base_pos[2]), 4),
            round(float(robot_x_dir[0]), 4), round(float(robot_x_dir[1]), 4), round(float(robot_x_dir[2]), 4)
        ]
        for i in range(4):
            row.extend([round(float(foot_xy[i, 0]), 4), round(float(foot_xy[i, 1]), 4)])
        row.extend([int(x) for x in contact])
        self.data_buffer.append(row)

    def save_to_csv(self):
        print(f"\n⏳ 正在将 {len(self.data_buffer)} 条数据写入CSV...")
        with open(self.file_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.data_buffer)
        print(f"✅ 数据保存成功: {self.file_path}")


# ====================== 主函数 ======================
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # ============== 1. 基础配置：单环境 ==============
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False        # 关闭观测噪声(便于测试)
    env_cfg.domain_rand.push_robots = False # 关闭外部推力扰动
    env_cfg.env.test = True

    # ============== 2. 复杂地形配置 (核心修复) ==============
    # 使用 trimesh 渲染真实物理三维地形，解决报错
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.selected = False       # 必须为False，使用默认网格地形生成器
    env_cfg.terrain.curriculum = False     # 关闭课程学习，直接出生在复杂地形
    
    # 缩小地形网格节省显存，2x2网格足够单机跑
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    
    # 定义地形比例 [平滑斜坡, 粗糙斜坡, 上楼梯, 下楼梯, 离散障碍物]
    # 这里设置为 100% 粗糙斜坡，如果想更复杂可以改为 [0.0, 0.5, 0.0, 0.0, 0.5]
    env_cfg.terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0]

    # ============== 3. 强制匀速直线运动 ==============
    env_cfg.commands.resampling_time = -1  # 禁用随机命令重采样
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5] # 锁定前向速度 0.5m/s
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0] # 锁定侧向速度
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0] # 锁定转向速度

    # ============== 4. 纯GPU物理引擎加速 ==============
    env_cfg.sim.use_gpu_pipeline = True
    env_cfg.sim.physx.use_gpu = True
    env_cfg.sim.device = 'cuda:0'

    # 打印运行参数
    print("=" * 40)
    print(f"✅ 运行模式: 纯GPU加速")
    print(f"✅ 运动指令: 匀速直线 0.5m/s")
    print(f"✅ 地形类型: 复杂地形 (粗糙斜坡)")
    print("=" * 40)

    # 初始化环境
    # 强制开启复杂地形的参数传递
    env_cfg.terrain.terrain_kwargs = {
        'type': 'rough',          # 地形类型
        'slope_threshold': 0.75,  # 允许的最大坡度
        'roughness': 0.2          # 关键参数：0.05到0.5之间，越大越崎岖
    }
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    print(f"👉 最终地形配置: {env.cfg.terrain.terrain_kwargs}") 
    obs = env.get_observations()
    
    # 加载策略模型
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # ====================== 准备数据采集 ======================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "foot_data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"foot_data_{timestamp}.csv")
    
    logger = DataLogger(csv_path)
    gym = env.gym
    sim = env.sim
    feet_indices = env.feet_indices
    start_time = time.time()

    # 运行总步数 (通常 10 个 episode 长度)
    total_steps = 10 * int(env.max_episode_length)

    try:
        for i in range(total_steps):
            # 持续强制覆盖速度指令 (双重保险)
            env.commands[:, 0] = 0.5
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            # 策略推理 & 环境步进
            actions = policy(obs.detach())
            obs, _, _, dones, _ = env.step(actions)

            # ====================== 提取记录数据 ======================
            # 1. 提取机器人基座姿态
            base_quat = env.base_quat[0].cpu().numpy()
            robot_x_dir = quat_to_x_dir(base_quat)
            base_pos = env.base_pos[0].cpu().numpy()

            # 2. 提取底层刚体状态获取真实足端 XY 坐标
            gym.refresh_rigid_body_state_tensor(sim)
            rigid_body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
            # 兼容性处理：重塑张量形状以匹配 [num_envs, num_bodies, 13]
            rigid_body_state = rigid_body_state.view(env.num_envs, -1, 13)
            foot_xy = rigid_body_state[0, feet_indices, :2].cpu().numpy()

            # 3. 提取足端接触状态 (标准逻辑：Z轴方向接触力 > 1牛顿认为发生接触)
            contact_forces = env.contact_forces[0, feet_indices, 2] # 提取 Z 轴力
            foot_contact = (contact_forces > 1.0).cpu().numpy().astype(int)

            # 写入内存 Buffer
            current_t = time.time() - start_time
            logger.add_record(current_t, base_pos, robot_x_dir, foot_xy, foot_contact)

            # 终端进度打印
            if i % 100 == 0:
                print(f"\r🏃 进度: {i:5d}/{total_steps} 步 | 运行时间: {current_t:.2f}s", end="")

            if dones[0]:
                obs = env.reset()

    except KeyboardInterrupt:
        print("\n⚠️ 收到中止信号，提前结束仿真...")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
    finally:
        # ======= 循环结束，一次性写入硬盘 =======
        logger.save_to_csv()

if __name__ == '__main__':
    check_gpu_memory()
    args = get_args()
    play(args)