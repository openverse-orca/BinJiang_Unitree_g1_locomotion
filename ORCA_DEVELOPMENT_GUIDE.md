# ORCA 开发 API 指南（从 0 到 1）

> 版本：1.2  
> 最后更新：2026-08-11  
> 适用范围：ORCA 仿真平台 + OrcaGym Python SDK  
> 支持系统：Windows 11 / Ubuntu 22.04 / Ubuntu 24.04

---

## 目录

1. [前言：ORCA 与 OrcaGym 是什么](#1-前言orca-与-orcagym-是什么)
2. [开发环境搭建](#2-开发环境搭建)
3. [第一个 ORCA 程序：连接、查询 Site、驱动关节](#3-第一个-orca-程序连接查询-site驱动关节)
4. [核心 API 详解](#4-核心-api-详解)
   - 4.1 创建环境
   - 4.2 查询 Site / Body / Joint
   - 4.3 驱动关节与推进仿真
   - 4.4 传感器与接触检测
   - 4.5 相机与渲染
5. [实战：三个典型任务](#5-实战三个典型任务)
   - 5.1 查询 Site 并导航
   - 5.2 查询关节状态
   - 5.3 驱动机械臂按压按钮
6. [调试与常见问题](#6-调试与常见问题)
7. [附录：API 速查表](#7-附录api-速查表)

---

## 1. 前言：ORCA 与 OrcaGym 是什么

**ORCA** 是松应科技开发的国产物理 AI 仿真平台，基于 MuJoCo 物理引擎，提供高保真机器人仿真、场景构建、数据采集与模型训练能力。

**OrcaGym** 是 ORCA 的 Python SDK，封装了与 ORCA 仿真器通信的 gRPC 接口，提供类似 Gymnasium 的 API，让开发者可以用熟悉的 Python 代码控制仿真中的机器人。

本指南基于 `OrcaLocomotion`、`OrcaManipulation` 仓库中的真实代码整理，力求可直接落地。

---

## 2. 开发环境搭建

### 2.1 安装 ORCA 仿真器

ORCA 仿真器通常以 ORCA Studio（企业版）或 ORCA Lab（开发者版）形式提供。安装后启动仿真并加载场景，确保：

- 场景中存在机器人模型（如 `unitree_humanoid_robot_1` 或 `g1_omnipicker`）。
- gRPC 服务端口默认开启在 `localhost:50051`。

### 2.2 创建 Python 环境

推荐使用 `orca-loco` conda 环境（Python 3.12）：

```bash
conda create -n orca-loco python=3.12 -y
conda activate orca-loco
```

### 2.3 安装 OrcaGym

OrcaGym 通常随 ORCA 安装包或独立仓库提供。如果已有源码：

```bash
cd /path/to/OrcaGym
pip install -e .
```

---

## 3. 第一个 ORCA 程序：连接、查询 Site、驱动关节

下面的脚本展示如何连接 ORCA、查询一个 Site 的世界坐标、查询机器人底盘位姿，并用 `set_joint_qpos` 小幅度移动机器人。

```python
"""first_orca.py — 第一个 ORCA 程序"""
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

# 1. 创建最简环境
class ProbeEnv(OrcaGymLocalEnv):
    _headless = True
    _render_mode = "none"
    _is_subenv = True

env = ProbeEnv(
    frame_skip=1,
    orcagym_addr="localhost:50051",
    agent_names=["unitree_humanoid_robot_1"],
    time_step=0.002,
)

try:
    # 2. 查询 Site 世界坐标
    site_name = "Static_start_site"
    site_dict = env.query_site_pos_and_quat([site_name])
    pos = np.asarray(site_dict[site_name]["xpos"])
    quat = np.asarray(site_dict[site_name]["xquat"])  # (w, x, y, z)
    print(f"[{site_name}] pos={pos}, quat={quat}")

    # 3. 查询机器人底盘位姿
    body_name = "unitree_humanoid_robot_1"
    xpos, xmat, xquat = env.get_body_xpos_xmat_xquat([body_name])
    print(f"[{body_name}] pos={xpos[body_name]}, quat={xquat[body_name]}")

    # 4. 查询 free joint 的 qpos
    free_joint = "unitree_humanoid_robot_1_free_joint"
    qpos = env.query_joint_qpos([free_joint])[free_joint]
    print(f"[{free_joint}] qpos={qpos}")  # [x, y, z, qw, qx, qy, qz]

    # 5. 让机器人沿 X 轴前进 0.5 米
    new_qpos = qpos.copy()
    new_qpos[0] += 0.5
    env.set_joint_qpos({free_joint: new_qpos.tolist()})
    env.mj_forward()  # 更新运动学与渲染

    # 6. 推进 1 个物理帧
    env.mj_step(nstep=1)

    # 7. 再次查询验证
    qpos2 = env.query_joint_qpos([free_joint])[free_joint]
    print(f"移动后 qpos={qpos2}")
finally:
    env.close()
```

**运行前确认**：

- ORCA 仿真已启动且场景加载完成。
- `localhost:50051` 可达。
- 机器人名称和关节名称与场景中一致（可通过第 4.1 节的场景扫描脚本获取）。

---

## 4. 核心 API 详解

### 4.1 创建环境

OrcaGym 提供 `OrcaGymLocalEnv` 和 `OrcaGymRemoteEnv`。大赛代码中最常用的是 `OrcaGymLocalEnv`。

#### 4.1.1 直接继承 OrcaGymLocalEnv

```python
from orca_gym.environment import OrcaGymLocalEnv

class MyTaskEnv(OrcaGymLocalEnv):
    def __init__(self, orcagym_addr, agent_names, **kwargs):
        super().__init__(
            frame_skip=5,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=0.002,
            render_mode="human",  # 或 "none"
            **kwargs,
        )
```

关键参数：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `orcagym_addr` | ORCA gRPC 地址 | `"localhost:50051"` |
| `agent_names` | 机器人实例名列表 | `["unitree_humanoid_robot_1"]` |
| `frame_skip` | 每步控制对应物理帧数 | `5` |
| `time_step` | MuJoCo 物理步长 | `0.002` |
| `render_mode` | 渲染模式 | `"human"` / `"none"` |

#### 4.1.2 使用 DataCollectionManager（G1 OmniPicker 任务推荐）

```python
from scene.scene_manager import SceneManager
from dataCollectionManager.data_collection_manager import DataCollectionManager

scene_manager = SceneManager("localhost:50051", config=level_config)

manager = DataCollectionManager(
    agent_name="g1_omnipicker",
    env_name="DataCollection",
    entry_point="envs.dataCollection.dataCollection_env:DataCollectionEnv",
    default_joint_values=build_default_joint_values(),
    obs_callback=_empty_obs_callback,
    env_index=0,
    device=None,
    scene_manager=scene_manager,
    frame_skip=5,
    orcagym_addr="localhost:50051",
)
env = manager.env
```

#### 4.1.3 扫描场景中可用名称

```python
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

class SceneProbeEnv(OrcaGymLocalEnv):
    _headless = True
    _render_mode = "none"
    _is_subenv = True

env = SceneProbeEnv(
    frame_skip=1,
    orcagym_addr="localhost:50051",
    agent_names=["SceneProbe"],
    time_step=0.002,
)

bodies = env.model.get_body_names()
joints = list(env.model.get_joint_dict().keys())
actuators = list(env.model.get_actuator_dict().keys())
sites = list(env.model.get_site_dict().keys())
sensors = list(env.model.get_sensor_dict().keys())

print("bodies:", bodies[:10])
print("joints:", joints[:10])
print("actuators:", actuators[:10])
print("sites:", sites[:10])
print("sensors:", sensors[:10])
env.close()
```

### 4.2 查询 Site / Body / Joint

#### 4.2.1 查询 Site 位置与姿态

```python
# 世界坐标系
site_dict = env.query_site_pos_and_quat(["Static_checkpoint1_site"])
pos = site_dict["Static_checkpoint1_site"]["xpos"]   # [x, y, z]
quat = site_dict["Static_checkpoint1_site"]["xquat"] # [w, x, y, z]

# 旋转矩阵格式
site_dict = env.query_site_pos_and_mat(["Static_checkpoint1_site"])
pos = site_dict["Static_checkpoint1_site"]["xpos"]
mat = site_dict["Static_checkpoint1_site"]["xmat"]   # 3x3

# base 坐标系（以机器人为参考）
base_body = env.body("unitree_humanoid_robot_1")
site_in_base = env.query_site_pos_and_quat_B(
    ["g1_omnipicker_l_hand_ee_site"],
    [base_body],
)
```

#### 4.2.2 查询 Body 位姿

```python
xpos, xmat, xquat = env.get_body_xpos_xmat_xquat(["unitree_humanoid_robot_1"])
pos = xpos["unitree_humanoid_robot_1"]      # [x, y, z]
quat = xquat["unitree_humanoid_robot_1"]    # [w, x, y, z]

# 只查位置
xpos_only = env.query_body_xpos(["unitree_humanoid_robot_1"])
```

#### 4.2.3 四元数转偏航角

```python
import numpy as np

def quat_to_yaw(q):
    """q: (w, x, y, z)"""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
```

#### 4.2.4 查询 Joint 状态

```python
# 关节位置
qpos = env.query_joint_qpos(["unitree_humanoid_robot_1_free_joint"])

# 关节速度
qvel = env.query_joint_qvel(["unitree_humanoid_robot_1_free_joint"])

# 关节在 qpos/qvel 数组中的偏移和长度
offsets = env.query_joint_offsets(joint_names)
lengths = env.query_joint_lengths(joint_names)
```

free joint 的 qpos 为 7 维 `[x, y, z, qw, qx, qy, qz]`，qvel 为 6 维 `[vx, vy, vz, wx, wy, wz]`。

#### 4.2.5 查询 Body 速度

```python
lin, ang = env.query_body_cvel(["unitree_humanoid_robot_1"])
print("线速度:", lin["unitree_humanoid_robot_1"])
print("角速度:", ang["unitree_humanoid_robot_1"])
```

### 4.3 驱动关节与推进仿真

#### 4.3.1 直接设置关节位置（瞬移/重置）

```python
free_joint = "unitree_humanoid_robot_1_free_joint"
new_qpos = [x, y, z, qw, qx, qy, qz]
new_qvel = [vx, vy, vz, wx, wy, wz]

env.set_joint_qpos({free_joint: new_qpos})
try:
    env.set_joint_qvel({free_joint: new_qvel})
except Exception:
    pass  # 某些后端不支持 set_joint_qvel
env.mj_forward()
```

#### 4.3.2 发送执行器控制指令

```python
# 准备控制缓冲区
env.prepare_control_buffer()

# 写入目标执行器控制值
env.ctrl[actuator_ids] = target_values

# 推进仿真
for _ in range(decimation):
    env.set_ctrl(env.ctrl)
    env.mj_step(nstep=env.frame_skip)

env.update_data()
```

#### 4.3.3 Gymnasium 风格 step

```python
action = np.zeros(env.action_space.shape)
obs, reward, terminated, truncated, info = env.step(action)
env.render()
```

### 4.4 传感器与接触检测

#### 4.4.1 查询传感器数据

```python
sensor_names = ["lf_foot_force", "rf_foot_force"]
data = env.query_sensor_data(sensor_names)
print(data["lf_foot_force"])
```

#### 4.4.2 查询接触对

```python
contacts = env.query_contact_simple()
for c in contacts:
    print(c["Geom1"], "<->", c["Geom2"])
```

### 4.5 相机与渲染

```python
# 更新可视化窗口
env.render()

# 启动内存流相机（视觉策略用）
env.begin_save_video("/path/to/trigger")
```

---

## 5. 实战：三个典型任务

### 5.1 查询 Site 并导航

场景：让 G1 机器人从起点出发，依次经过检查点 1、检查点 2，到达终点。

```python
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

class NavEnv(OrcaGymLocalEnv):
    _headless = True
    _render_mode = "none"
    _is_subenv = True

env = NavEnv(
    frame_skip=1,
    orcagym_addr="localhost:50051",
    agent_names=["unitree_humanoid_robot_1"],
    time_step=0.002,
)

free_joint = "unitree_humanoid_robot_1_free_joint"
robot_body = "unitree_humanoid_robot_1"

def get_site_pos(name):
    d = env.query_site_pos_and_quat([name])
    return np.asarray(d[name]["xpos"])

def get_robot_xyyaw():
    xpos, _, xquat = env.get_body_xpos_xmat_xquat([robot_body])
    pos = np.asarray(xpos[robot_body])
    q = np.asarray(xquat[robot_body])
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pos[:2], yaw

try:
    waypoints = ["Static_start_site", "Static_checkpoint1_site", "Static_checkpoint2_site", "Static_end_site"]
    for wp in waypoints:
        target = get_site_pos(wp)
        for _ in range(1000):
            pos, yaw = get_robot_xyyaw()
            delta = target[:2] - pos
            dist = np.linalg.norm(delta)
            if dist < 0.5:
                print(f"到达 {wp}")
                break

            target_yaw = np.arctan2(delta[1], delta[0])
            yaw_err = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

            # 简化控制：直接积分移动底盘（实际应结合 HEFT 步态）
            qpos = env.query_joint_qpos([free_joint])[free_joint].copy()
            qpos[0] += 0.02 * np.cos(yaw)
            qpos[1] += 0.02 * np.sin(yaw)
            qpos[6] += 0.05 * yaw_err  # 简单转向
            env.set_joint_qpos({free_joint: qpos.tolist()})
            env.mj_forward()
            env.mj_step(nstep=1)
finally:
    env.close()
```

> 实际导航应结合 HEFT 策略生成自然步态，参见 `run_task1_heft_navigation.py`。

### 5.2 查询关节状态

```python
joints = [
    "unitree_humanoid_robot_1_left_hip_pitch_joint",
    "unitree_humanoid_robot_1_right_hip_pitch_joint",
]
qpos = env.query_joint_qpos(joints)
qvel = env.query_joint_qvel(joints)
for j in joints:
    print(f"{j}: qpos={qpos[j]}, qvel={qvel[j]}")
```

### 5.3 驱动机械臂按压按钮

```python
# 伪代码：将机械臂末端移动到按钮 Site 附近
button_site = "Group_Static_ElectricalCabinet_button01_site"
button_pos = env.query_site_pos_and_quat([button_site])[button_site]["xpos"]

# 使用逆运动学或关节插值让末端接近按钮
# ... IK / joint interpolation ...

# 判定是否按压成功
ee_site = "g1_omnipicker_l_hand_ee_site"
ee_pos = env.query_site_pos_and_quat([ee_site])[ee_site]["xpos"]
distance = np.linalg.norm(np.asarray(ee_pos) - np.asarray(button_pos))
print(f"末端到按钮距离: {distance:.3f} m")
```

---

## 6. 调试与常见问题

### 6.1 ORCA 连接失败

- 确认 ORCA 仿真已启动且场景加载完成。
- 确认 gRPC 端口 `localhost:50051` 可达。
- 检查防火墙是否拦截。

### 6.2 Site / Body / Joint 名称不存在

- 用第 4.1.3 节的场景扫描脚本打印所有可用名称。
- 注意机器人实例可能有前缀，如 `unitree_humanoid_robot_1_free_joint`。

### 6.3 设置 qpos 后画面没更新

- 调用 `env.mj_forward()` 刷新运动学和渲染。

### 6.4 set_joint_qvel 报错

- 某些后端不支持 `set_joint_qvel`，用 try/except 包裹并忽略。

### 6.5 PermissionError on .lock file

Windows 下 OrcaGym 缓存目录权限不足时，可以切换到用户有写入权限的临时目录：

```powershell
$env:USERPROFILE = "$env:TEMP\orcahome"
$env:HOME = $env:USERPROFILE
```

Ubuntu 下可类似设置：

```bash
export HOME=/tmp/orcahome
```

---

## 7. 附录：API 速查表

| 功能 | API | 返回值/作用 |
|------|-----|------------|
| 创建本地环境 | `OrcaGymLocalEnv(...)` | 环境实例 |
| 扫描场景 | `env.model.get_body_names()` / `get_joint_dict()` / `get_site_dict()` | 名称列表/字典 |
| 查询 Site 世界位姿 | `env.query_site_pos_and_quat(names)` | `{name: {xpos, xquat}}` |
| 查询 Site 旋转矩阵 | `env.query_site_pos_and_mat(names)` | `{name: {xpos, xmat}}` |
| 查询 Site 在 base 下 | `env.query_site_pos_and_quat_B(names, base_bodies)` | base 坐标系位姿 |
| 查询 Body 世界位姿 | `env.get_body_xpos_xmat_xquat(names)` | `(xpos, xmat, xquat)` |
| 查询 Body 速度 | `env.query_body_cvel(names)` | `(lin_dict, ang_dict)` |
| 查询 Joint 位置 | `env.query_joint_qpos(names)` | `{name: qpos}` |
| 查询 Joint 速度 | `env.query_joint_qvel(names)` | `{name: qvel}` |
| 设置 Joint 位置 | `env.set_joint_qpos({name: qpos})` | 瞬移/重置 |
| 设置 Joint 速度 | `env.set_joint_qvel({name: qvel})` | 部分后端支持 |
| 发送控制 | `env.set_ctrl(ctrl)` / `env.prepare_control_buffer()` | 驱动执行器 |
| 推进仿真 | `env.mj_step(nstep=N)` | 推进 N 物理帧 |
| 刷新运动学 | `env.mj_forward()` | 更新渲染状态 |
| Gym step | `env.step(action)` | `(obs, reward, terminated, truncated, info)` |
| 传感器 | `env.query_sensor_data(names)` | `{name: value}` |
| 接触检测 | `env.query_contact_simple()` | 接触对列表 |
| 渲染 | `env.render()` | 更新窗口 |

---

> 本指南代码片段均基于 `OrcaLocomotion`、`OrcaManipulation` 仓库中的真实实现整理。实际开发时请根据具体场景和机器人模型调整名称与参数。
