"""
Task1 移动巡检 - HEFT 自然行走版本

用 HEFT 策略驱动宇树 G1 双足机器人按 start → checkpoint1 → checkpoint2 → end 的顺序移动。

运行前请确保：
1. ORCA 已加载包含 unitree_humanoid_robot_1 和 Static_start_site / Static_checkpoint1_site / Static_checkpoint2_site / Static_end_site 的场景

支持系统：Windows 11 / Ubuntu 22.04 / Ubuntu 24.04

用法：
    conda run -n orca-loco python run_task1_heft_navigation.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

# 仅注入 OrcaLocomotion 根目录，用于导入 orca_rl
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orca_rl.utils import (
    apply_remote_override,
    check_orcagym_addresses,
    ensure_project_root_on_path,
    explain_missing_runtime_dependency,
)

ensure_project_root_on_path()

DEFAULT_NAV_CONFIG = PROJECT_ROOT / "nav_task1.yaml"

DEFAULT_POLICY = PROJECT_ROOT / "checkpoints/heft/G1_PMG/policy.onnx"
DEFAULT_MOTION_DIR = PROJECT_ROOT / "assets/heft/recorded_commands"
DEFAULT_WALK_MOTION_DIR = PROJECT_ROOT / "assets/heft/motions"

# 导航默认参数（如需调优可直接修改下面常量）
FORWARD_SPEED = 0.20
YAW_CMD = 0.35
ALIGN_THRESHOLD = 0.25
MAX_STEPS_PER_WP = 10000
TRANSITION_S = 0.4
ONNX_THREADS = 4

# 根据 --robot-id 覆盖导航配置里的机器人 body/joint 名。
ROBOT_OVERRIDES: dict[str, dict[str, Any]] = {
    "g1_pick": {
        "base_body": "g1_pick_pelvis",
        "free_joint": "g1_pick_floating_base_joint",
        "ee_site": "g1_pick_left_palm",
        "ee_site_right": "g1_pick_right_palm",
        "finger_a": "g1_pick_left_hand_thumb_0_joint",
        "finger_b": "g1_pick_left_hand_index_0_joint",
    },
    "unitree_g1": {
        "base_body": "unitree_humanoid_robot_1_pelvis",
        "free_joint": "unitree_humanoid_robot_1_floating_base_joint",
        "ee_site": "unitree_humanoid_robot_1_left_palm",
        "ee_site_right": "unitree_humanoid_robot_1_right_palm",
        "finger_a": "unitree_humanoid_robot_1_left_hand_thumb_0_joint",
        "finger_b": "unitree_humanoid_robot_1_left_hand_index_0_joint",
    },
}


def load_nav_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_heft_scene(task_cfg: dict[str, Any]) -> None:
    """沿用 play_g1_heft_velocity 的场景配置。"""
    task_cfg["num_envs"] = 1
    task_cfg.setdefault("sim", {}).update(
        render_mode="human",
        headless=False,
        unitree_play_global_settings=True,
    )
    task_cfg.setdefault("episode", {})["length_s"] = 1.0e9
    task_cfg.setdefault("observations", {})["add_noise"] = False
    task_cfg["curriculum"] = {}
    randomization = task_cfg.setdefault("randomization", {})
    randomization.update(enabled=False, max_action_delay_steps=0)
    events = task_cfg.setdefault("events", {})
    for name in tuple(events):
        if name.startswith("randomize_") or name == "push_robot":
            events.pop(name, None)
    terrain = task_cfg.get("terrain")
    if isinstance(terrain, dict) and terrain.get("terrain_type") == "plane":
        terrain["physics_enabled"] = False


def get_body_pos_yaw(task, body_name: str):
    """从 ORCA 查询指定 body 的 XY 位置与偏航角。"""
    xpos, _, xquat = task.get_body_xpos_xmat_xquat([body_name])
    pos = np.asarray(xpos, dtype=np.float64).reshape(3)
    quat = np.asarray(xquat, dtype=np.float64).reshape(4)  # (w, x, y, z)
    w, x, y, z = quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return pos, float(yaw)


def get_site_pos(task, site_name: str) -> np.ndarray:
    """从 ORCA 查询指定 site 的世界坐标。"""
    site_dict = task.query_site_pos_and_quat([site_name])
    return np.asarray(site_dict[site_name]["xpos"], dtype=np.float64).reshape(3)


def _quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def build_route(
    task,
    waypoints: list[dict[str, Any]],
    arrival_radius: float = 0.6,
) -> list[dict[str, Any]]:
    """根据 site waypoints 构建实际导航路点序列。"""
    route: list[dict[str, Any]] = []
    for wp in waypoints:
        site_name = wp["site_name"]
        radius = float(wp.get("radius", arrival_radius))
        target_pos = get_site_pos(task, site_name)
        route.append({
            "site_name": site_name,
            "radius": radius,
            "target_pos": target_pos,
        })
    return route


def run_navigation_loop(
    env,
    bridge,
    route: list[dict[str, Any]],
    robot_cfg: dict[str, Any],
) -> dict[str, Any]:
    """HEFT + 底盘 free joint 位姿积分，依次到达路点。"""
    task = env.tasks[0]
    base_body = robot_cfg["base_body"]
    free_joint = robot_cfg.get("free_joint", "unitree_humanoid_robot_1_floating_base_joint")
    dt = float(task.control_dt)

    reached: list[str] = []
    total_steps = 0
    stand_cmd = np.zeros((env.num_envs, 3), dtype=np.float64)

    for wp in route:
        site_name = wp["site_name"]
        radius = float(wp["radius"])
        target_pos = np.asarray(wp["target_pos"], dtype=np.float64).reshape(3)
        print(f"\n[nav] 目标: {site_name} @ ({target_pos[0]:.2f}, {target_pos[1]:.2f})")

        for step in range(MAX_STEPS_PER_WP):
            current_pos, current_yaw = get_body_pos_yaw(task, base_body)
            dx = float(target_pos[0] - current_pos[0])
            dy = float(target_pos[1] - current_pos[1])
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

            if dist < radius:
                print(f"[nav] 到达 {site_name} (dist={dist:.2f}m)")
                reached.append(site_name)
                break

            if abs(yaw_err) > ALIGN_THRESHOLD:
                cmd = np.array([0.0, 0.0, math.copysign(YAW_CMD, yaw_err)], dtype=np.float64)
                bridge.set_commands(np.repeat(cmd.reshape(1, 3), env.num_envs, axis=0))
                actions = bridge.act()
                bridge.step(actions)
            else:
                cmd = np.array([FORWARD_SPEED, 0.0, 0.0], dtype=np.float64)
                bridge.set_commands(np.repeat(cmd.reshape(1, 3), env.num_envs, axis=0))
                actions = bridge.act()

                fj_qpos = task.query_joint_qpos([free_joint])[free_joint]
                fj_qpos = np.asarray(fj_qpos).flatten().copy()
                fj_pos = fj_qpos[:3]
                fj_yaw = _quat_to_yaw(fj_qpos[3:7])

                wz_corr = float(np.clip(yaw_err * 1.5, -0.2, 0.2))
                new_yaw = fj_yaw + wz_corr * dt
                step_dist = FORWARD_SPEED * dt
                new_x = fj_pos[0] + step_dist * math.cos(target_yaw)
                new_y = fj_pos[1] + step_dist * math.sin(target_yaw)
                half_yaw = new_yaw / 2.0
                new_qpos = [
                    float(new_x), float(new_y), float(fj_pos[2]),
                    math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw),
                ]
                new_qvel = [
                    FORWARD_SPEED * math.cos(target_yaw),
                    FORWARD_SPEED * math.sin(target_yaw),
                    0.0, 0.0, 0.0, wz_corr,
                ]
                task.set_joint_qpos({free_joint: new_qpos})
                try:
                    task.set_joint_qvel({free_joint: new_qvel})
                except Exception:
                    pass
                task.mj_forward()
                bridge.step(actions)

            total_steps += 1
            if step % 25 == 0:
                print(
                    f"  [{step:3d}] pos=({current_pos[0]:.2f},{current_pos[1]:.2f}) "
                    f"yaw={math.degrees(current_yaw):5.1f}° dist={dist:.2f}m "
                    f"err={math.degrees(yaw_err):4.1f}°"
                )
        else:
            print(f"[nav] 超时未到达 {site_name}")

    # 最后停止：HEFT 站立
    bridge.set_commands(stand_cmd)
    for _ in range(30):
        actions = bridge.act()
        bridge.step(actions)

    return {
        "reached": reached,
        "total_steps": total_steps,
        "elapsed_s": total_steps * dt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task1 HEFT navigation")
    parser.add_argument("--robot-id", default="g1_pick", help="机器人形态 ID（默认: g1_pick）")
    parser.add_argument("--nav-config", default=str(DEFAULT_NAV_CONFIG), help="导航 YAML 配置路径")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="HEFT policy.onnx 路径")
    parser.add_argument("--motion-dir", default=str(DEFAULT_MOTION_DIR), help="HEFT motion 目录")
    parser.add_argument("--walk-motion-dir", default=str(DEFAULT_WALK_MOTION_DIR), help="HEFT walk motion 目录")
    parser.add_argument("--remote", default=None, help="ORCA gRPC 地址，默认 localhost:50051")
    args = parser.parse_args()

    nav_cfg = load_nav_config(Path(args.nav_config))
    robot_cfg = nav_cfg.get("robot", {})
    waypoints = nav_cfg.get("waypoints", [])
    if not waypoints:
        raise ValueError("导航配置中缺少 waypoints")

    # 根据 --robot-id 覆盖 nav_config 中的机器人配置。
    override = ROBOT_OVERRIDES.get(args.robot_id)
    if override:
        robot_cfg.update(override)
        print(f"[heft-task1] 根据 --robot-id={args.robot_id} 覆盖机器人配置: {override}")

    try:
        from orca_rl.heft_env import make_heft_env
        from orca_rl.rsl_env.heft_policy import HeftG1RecordedCommandBridge
        from orca_rl.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
    except ImportError as exc:
        raise explain_missing_runtime_dependency(exc) from exc

    task_cfg = unitree_g1_flat_env_cfg(play=True).to_dict()
    apply_remote_override(task_cfg, args.remote)
    configure_heft_scene(task_cfg)
    check_orcagym_addresses(task_cfg)

    env = make_heft_env(task_cfg)

    try:
        print(
            f"[heft-task1] robot=G1+Dex3-1 envs={env.num_envs} "
            f"remote={task_cfg['orcagym_addresses'][0]}"
        )
        dt = float(env.tasks[0].control_dt)
        bridge = HeftG1RecordedCommandBridge(
            env,
            policy_path=args.policy,
            motion_dir=args.motion_dir,
            walk_motion_dir=args.walk_motion_dir,
            transition_steps=max(0, int(round(TRANSITION_S / dt))),
            onnx_threads=ONNX_THREADS,
        )
        bridge.reset(reset_env=True)

        bridge.set_commands(np.zeros((env.num_envs, 3), dtype=np.float64))
        for _ in range(30):
            actions = bridge.act()
            bridge.step(actions)

        route = build_route(env.tasks[0], waypoints, arrival_radius=0.6)

        stats = run_navigation_loop(env, bridge, route, robot_cfg)
        print(f"\n[heft-task1] 导航完成: {stats}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
