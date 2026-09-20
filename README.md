# Orca HEFT G1 

本仓库用于在 OrcaLab 中运行（回放）上游 HEFT 项目训练的 G1 PMG 行走策略，仅包含推理所需的代码、ONNX 策略、动作片段与安装脚本，不包含训练部分。

策略动作维度固定为 29，只控制 G1 本体。

---

## 1. 环境要求

- 操作系统：Windows 11 / Ubuntu 22.04 / Ubuntu 24.04
- Python：3.12 及以上
- OrcaLab / OrcaStudio：26.7.1（兼容 26.6.3 及以上）
- 建议：使用 `orca-loco` conda 环境

---

## 2. 安装

### 2.1 克隆仓库

```bash
git clone --branch master https://github.com/openverse-orca/BinJiang_Unitree_g1_locomotion.git
cd BinJiang_Unitree_g1_locomotion
```

### 2.2 安装 HEFT 运行环境

默认使用 `orca-loco` conda 环境（Python 3.12），首次使用需先创建：

```bash
conda create -n orca-loco python=3.12 -y
```

在仓库根目录执行：

```bash
conda activate orca-loco
```

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-deps -e .
```

安装完成后校验资产与运行时依赖：

```bash
./scripts/check_heft_install.sh --runtime
```

`check_heft_install.sh` 会自动检测 `orca-loco` conda 环境；如需使用其他 Python，可设置 `ORCA_HEFT_PYTHON`：

```bash
ORCA_HEFT_PYTHON=/path/to/python3.12 ./scripts/check_heft_install.sh --runtime
```

### 2.3 校验安装

```bash
./scripts/check_heft_install.sh --runtime
```

所有发布资产都有 SHA-256 校验。

---

## 3. 启动 ORCA 并加载场景

1. 打开 OrcaLab / OrcaStudio。
2. 打开仓库根目录的布局文件 [`g1_pick_layout.json`](g1_pick_layout.json)，会自动加载滨江比赛场景（含配电箱等互动资产）和 `g1_pick` 机器人。
3. 确认场景中包含以下路点 site（按 start → checkpoint1 → checkpoint2 → end 顺序）：
   - `Static_start_site`
   - `Static_checkpoint1_site`
   - `Static_checkpoint2_site`
   - `Static_end_site`
4. 在 ORCA 中选择「外部仿真程序」模式，确认机器人处于可控制状态。

如果场景中找不到 G1 机器人，脚本会报错：

```text
ValueError: Cannot find robot model in scene: G1
```

---

## 4. 基础运行：键盘遥控（可选验证）

可先用键盘验证 HEFT 策略能正常驱动机器人。

```bash
./play_g1_heft.sh
```

非默认服务地址：

```bash
./play_g1_heft.sh --remote 127.0.0.1:50051
```

| 按键 | 动作 |
| --- | --- |
| `W/S`、上下方向键 | 前进/后退 |
| `A/D`、左右方向键 | 左移/右移 |
| `Z/C` | 左转/右转 |
| `F1/F2/F3` | HEFT walk1/walk2/walk3 |
| 空格 | 站立 |
| `R` | 复位 |
| `Q`、`Esc` | 退出 |

SSH / 无全局键盘权限时使用：

```bash
./play_g1_heft.sh --keyboard-backend terminal
```

固定命令联调：

```bash
./play_g1_heft.sh --keyboard-backend none --lin-vel-x 0.5 --seconds 20
```

兼容入口 `./play_g1_heft_velocity.sh` 保留。

---

## 5. Task1 移动巡检

### 5.1 运行导航

**Windows（PowerShell）：**

```powershell
conda run -n orca-loco python run_task1_heft_navigation.py
```

**Ubuntu 22.04/24.04（Bash）：**

```bash
conda run -n orca-loco python run_task1_heft_navigation.py
```

预期输出：

```text
[nav] 到达 Static_start_site (dist=0.57m)
[nav] 到达 Static_checkpoint1_site (dist=0.55m)
[nav] 到达 Static_checkpoint2_site (dist=0.60m)
[nav] 到达 Static_end_site (dist=0.58m)
[heft-task1] 导航完成: {'reached': [...], 'total_steps': 5100, 'elapsed_s': 102.0}
```

> 当前默认速度较慢（`FORWARD_SPEED = 0.20 m/s`），以保证双足行走自然、不碰撞障碍物。全程约 20 m，完整导航约需 2 分钟，请耐心等待。

---

## 6. 常见问题与排查

### 6.1 HEFT 机器人原地踏步、不前进

HEFT 录制的 `heft_forward` 动作在 ORCA 中表现为原地踏步。本脚本通过 free joint 位姿积分驱动底盘前进，因此这是预期设计。

### 6.2 某个检查点超时未到达

查看日志中 `dist` 是否持续减小。若接近目标但步数耗尽，可增大 `run_navigation_loop` 中的 `MAX_STEPS_PER_WP`；若转向反复摆动，可增大 `ALIGN_THRESHOLD` 或降低 `YAW_CMD`。

---

## 7. 验证流程清单

- [ ] ORCA 已启动并加载包含 G1 与 start/checkpoint1/checkpoint2/end 路点 site 的场景
- [ ] `orca-loco` conda 环境可用
- [ ] 运行 Task1 导航验证四个路点均可到达

---

## 8. HEFT 引用

本运行包内的 G1 PMG 策略和 `walk1/walk2/walk3` 动作源自 [Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) 的 `sim2real` 分支，固定版本为 [`0d5ba31e33397f3543d350d98b637e26d92f470a`](https://github.com/Axellwppr/motion_tracking/commit/0d5ba31e33397f3543d350d98b637e26d92f470a)，按 MIT License 使用；Copyright (c) 2026 Axell。

完整部署与排障另见 [`docs/HEFT_DEX3.md`](docs/HEFT_DEX3.md)。
