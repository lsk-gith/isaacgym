"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


1080 balls of solitude
-------------------------
Demonstrates the use of collision filtering to limit collisions to actors within an environment,
simulate all collisions including between actors in different environments, or simulate no collisions between
actors - they will still collide with the ground plane.

Modes can be set via command line arguments:
    --no_collisions to have no actors colide with other actors
    --all_collisions to have all actors, even those from different environments, collide

Press 'R' to reset the  simulation
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# configure sim
sim_params = gymapi.SimParams()
'''
sim_params: 模拟参数配置
字段	                             作用	                                             常用配置示例
dt	                             仿真的时间步长（Timestep），即每步模拟的固定时长	         sim_params.dt = 1 / 60 表示每秒60步
substeps	                     将一步仿真细分为多个子步进行物理计算，有助于提升稳定性	     sim_params.substeps = 2 表示将1步时间再细分2次迭代
up_axis	                         定义世界坐标系中的"向上"方向轴	                         sim_params.up_axis = gymapi.UP_AXIS_Z 表示Z轴向上
gravity	                         设置场景中的重力加速度向量	                             sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
use_gpu_pipeline	             核心开关：是否启用GPU加速的仿真与渲染管线	             sim_params.use_gpu_pipeline = True 开启后即可使用张量API
physx.use_gpu	                 PhysX专属：配置物理求解器是否使用GPU加速	             sim_params.physx.use_gpu = True 配合上面的总开关启用
physx.solver_type	             PhysX专属：选择关节求解器类型（1 为TGS，更稳定）	         sim_params.physx.solver_type = 1
physx.num_position_iterations	 PhysX专属：位置求解迭代次数，越高接触越稳定，但计算量更大	 sim_params.physx.num_position_iterations = 6
physx.num_velocity_iterations	 PhysX专属：速度求解迭代次数	                         sim_params.physx.num_velocity_iterations = 1
physx.contact_offset	         PhysX专属：为提升性能设置的接触检测距离阈值	             sim_params.physx.contact_offset = 0.01
sim_params.physx.num_threads     PhysX 引擎用于执行物理计算的 CPU 线程数                  default=0 通常的含义是“让 PhysX 自动选择最优线程数” 
                                                                                         如果use_gpu_pipeline = True即启用 GPU 加速管线，改参数会被忽略，


'''
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
'''
应该是你构建仿真世界的第一步。它的核心作用是创建一个包含物理与图形上下文的“仿真世界骨架”，为后续加载物体、创建环境和执行模拟提供基础

compute_device_id	显卡序号 例如:'cuda:0'，选择GPU进行物理模拟
graphics_device_id	渲染显卡序号，例如:0 在第一张卡上渲染，如果:-1不需要渲染，例如纯RL训练场景
gymapi.SIM_PHYSX	指定要使用的物理引擎类型（SIM_PHYSX:NVIDIA的PhysX引擎或者SIM_FLEX:Flex引擎基本弃用）
sim_params	        其他模拟参数 一个 gymapi.SimParams() 类型的对象，用于精细配置仿真的各项参数，包括时间步长、求解器设置、重力方向等
'''
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load ball asset
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# set up the env grid
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # create ball pyramid
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    n = 4
    radius = 0.2
    ball_spacing = 2.5 * radius
    min_coord = -0.5 * (n - 1) * ball_spacing
    y = min_coord+4
    while n > 0:
        z = min_coord
        for j in range(n):
            x = min_coord
            for k in range(n):
                pose.p = gymapi.Vec3(x, 1.5 + y, z)

                # Set up collision filtering.
                if args.all_collisions:
                    # Everything should collide.
                    # Put all actors in the same group, with filtering mask set to 0 (no filtering).
                    collision_group = 0
                    collision_filter = 0

                elif args.no_collisions:
                    # Nothing should collide.
                    # Use identical filtering masks for all actors to filter collisions between them.
                    # Group assignment doesn't matter in this case.
                    # Alternative would be to put each actor in a different group.
                    collision_group = 0
                    collision_filter = 1

                else:
                    # Balls in the same env should collide, but not with balls from different envs.
                    # Use one group per env, and filtering masks set to 0.
                    collision_group = i
                    collision_filter = 0

                ahandle = gym.create_actor(env, asset, pose, None, collision_group, collision_filter)
                gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

                x += ball_spacing
            z += ball_spacing
        y += ball_spacing
        n -= 1
        min_coord = -0.5 * (n - 1) * ball_spacing

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
