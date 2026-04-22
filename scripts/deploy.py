import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from envs.task1_pick_env import PickEnv, JOINT_NAMES, JOINT_LIMITS, N_SUBSTEPS

BIN_CENTERS = {
    1: np.array([0.30, -0.20, 0.42]),
    2: np.array([0.30,  0.20, 0.42]),
}
BIN_XY_RADIUS = 0.11
LIFT_HEIGHT = 0.18
PLACE_HEIGHT = 0.055
IK_STEP_SIZE = 0.4
IK_DAMPING = 1e-4
IK_TOL = 0.012
IK_MAX_STEPS = 500

ALGO = "ppo"
STAGE = 2
EPISODES = 5
STEP_DELAY = 0.02


def _check_bin_collision_config(env):
    geom_names = [
        "bin_can_floor", "bin_can_wall_back",
        "bin_carton_floor", "bin_carton_wall_back",
        "obj_can_geom", "obj_milk_geom",
    ]
    print("Collision mask check")
    for gname in geom_names:
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            print(f"warning: geom not found: {gname}")
            continue
        ct = int(env.model.geom_contype[gid])
        ca = int(env.model.geom_conaffinity[gid])
        tag = "ok" if (ct != 0 and ca != 0) else "warning"
        print(f"{tag}: {gname} contype={ct} conaffinity={ca}")


def _default_model(algo: str, stage: int) -> str:
    return os.path.join(PROJECT_ROOT, "models", algo, f"stage{stage}", "best", "best_model.zip")


def _dof_addrs(model) -> np.ndarray:
    return np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in JOINT_NAMES
    ], dtype=np.intp)


def _ik_move_to(env, dof_addr, target_pos, viewer=None, step_delay=0.02, grip_closed=True):
    jacp = np.zeros((3, env.model.nv))
    grip_val = 0.0 if grip_closed else 255.0

    for _ in range(IK_MAX_STEPS):
        mujoco.mj_forward(env.model, env.data)
        err  = target_pos - env.data.site_xpos[env._eef_site_id]
        dist = float(np.linalg.norm(err))
        if dist < IK_TOL:
            break

        jacp[:] = 0.0
        mujoco.mj_jacSite(env.model, env.data, jacp, None, env._eef_site_id)
        J  = jacp[:, dof_addr]
        dq = IK_STEP_SIZE * J.T @ np.linalg.solve(
            J @ J.T + IK_DAMPING * np.eye(3), err)
        dq = np.clip(dq, -0.05, 0.05)

        q = np.clip(
            env.data.qpos[env._qpos_addrs] + dq,
            JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        env.data.qpos[env._qpos_addrs]  = q
        env.data.ctrl[env._arm_act_ids] = q
        env.data.ctrl[env._grip_act_id] = grip_val

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(env.model, env.data)

        if viewer is not None:
            viewer.sync()
            time.sleep(step_delay)

    return dist


def _close_gripper_visual(env, viewer=None, hold_steps=60, step_delay=0.02):
    env.data.ctrl[env._grip_act_id] = 0.0
    for _ in range(hold_steps * N_SUBSTEPS):
        mujoco.mj_step(env.model, env.data)
    if viewer is not None:
        viewer.sync()
        time.sleep(step_delay * hold_steps * 0.3)


def _open_gripper(env, viewer=None, hold_steps=60, step_delay=0.02):
    env.data.ctrl[env._grip_act_id] = 255.0
    for _ in range(hold_steps * N_SUBSTEPS):
        mujoco.mj_step(env.model, env.data)
    if viewer is not None:
        viewer.sync()
        time.sleep(step_delay * hold_steps * 0.5)


def _ik_move_carrying(env, dof_addr, target_pos, active_cls, grasp_offset,
                      viewer=None, step_delay=0.02):
    jacp = np.zeros((3, env.model.nv))
    qpos_addr = env._obj_qpos_addrs[active_cls]
    qvel_addr = env._obj_qvel_addrs[active_cls]

    for _ in range(IK_MAX_STEPS):
        mujoco.mj_forward(env.model, env.data)
        err  = target_pos - env.data.site_xpos[env._eef_site_id]
        dist = float(np.linalg.norm(err))
        if dist < IK_TOL:
            break

        jacp[:] = 0.0
        mujoco.mj_jacSite(env.model, env.data, jacp, None, env._eef_site_id)
        J  = jacp[:, dof_addr]
        dq = IK_STEP_SIZE * J.T @ np.linalg.solve(
            J @ J.T + IK_DAMPING * np.eye(3), err)
        dq = np.clip(dq, -0.05, 0.05)

        q = np.clip(
            env.data.qpos[env._qpos_addrs] + dq,
            JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        env.data.qpos[env._qpos_addrs]  = q
        env.data.ctrl[env._arm_act_ids] = q
        env.data.ctrl[env._grip_act_id] = 0.0

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(env.model, env.data)

        eef_pos = env.data.site_xpos[env._eef_site_id].copy()
        env.data.qpos[qpos_addr:qpos_addr + 3] = eef_pos + grasp_offset
        env.data.qvel[qvel_addr:qvel_addr + 6] = 0.0

        if viewer is not None:
            viewer.sync()
            time.sleep(step_delay)

    return dist


def _placed_correctly(obj_pos: np.ndarray, cls: int) -> bool:
    bc = BIN_CENTERS.get(cls)
    if bc is None:
        return False
    return (abs(obj_pos[0] - bc[0]) < BIN_XY_RADIUS and
            abs(obj_pos[1] - bc[1]) < BIN_XY_RADIUS)


def main():
    algo = ALGO
    stage = STAGE
    episodes = EPISODES
    step_delay = STEP_DELAY

    model_path = _default_model(algo, stage)
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        sys.exit(1)

    from stable_baselines3 import PPO, SAC
    algo_cls = {"ppo": PPO, "sac": SAC}[algo]
    print(f"Loading {algo.upper()} stage {stage} checkpoint: {model_path}")
    policy = algo_cls.load(model_path)

    stage2 = (stage == 2)
    env = PickEnv(
        active_object   = None if stage2 else 1,
        conveyor_speed  = 0.05,
        randomize_speed = stage2,
        object_classes  = [1, 2] if stage2 else None,
        use_gt_pos      = False,
        render_mode     = "rgb_array",
    )

    print("Using YOLO perception from camera localization")

    _phantom_id   = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "phantom_joint")
    _phantom_qadr = env.model.jnt_qposadr[_phantom_id]

    def _update_phantom():
        if env._perception is not None and env._perception.last_pos_3d is not None:
            env.data.qpos[_phantom_qadr:_phantom_qadr + 3] = env._perception.last_pos_3d
            env.data.qpos[_phantom_qadr + 3] = 1.0
            env.data.qpos[_phantom_qadr + 4:_phantom_qadr + 7] = 0.0
            mujoco.mj_forward(env.model, env.data)

    dof_addr = _dof_addrs(env.model)
    _check_bin_collision_config(env)

    CLASS_NAMES   = {1: "soda_can", 2: "milk_carton"}
    grasp_total   = 0
    place_total   = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(episodes):
            obs, _      = env.reset()
            active_cls  = env._active_idx
            total_rew   = 0.0
            n_steps     = 0
            done        = False
            grasped     = False

            while viewer.is_running() and not done:
                action, _ = policy.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = env.step(action)
                total_rew += rew
                n_steps   += 1
                done       = terminated or truncated
                if info.get("success", False):
                    grasped = True

                _update_phantom()
                viewer.sync()
                time.sleep(step_delay)

            if not viewer.is_running():
                break

            placed = False

            if grasped and active_cls in BIN_CENTERS:
                grasp_total += 1
                obj_pos      = env.data.xpos[env._obj_ids[active_cls]].copy()
                grasp_offset = obj_pos - env.data.site_xpos[env._eef_site_id].copy()
                env.data.ctrl[env._grip_act_id] = 0.0
                for _ in range(40):
                    for _ in range(N_SUBSTEPS):
                        mujoco.mj_step(env.model, env.data)
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(step_delay)

                lift_pos      = env.data.site_xpos[env._eef_site_id].copy()
                lift_pos[2]  += LIFT_HEIGHT
                _ik_move_carrying(env, dof_addr, lift_pos, active_cls,
                                  grasp_offset, viewer, step_delay)

                bin_c         = BIN_CENTERS[active_cls].copy()
                above_bin     = bin_c.copy()
                above_bin[2]  = lift_pos[2]
                _ik_move_carrying(env, dof_addr, above_bin, active_cls,
                                  grasp_offset, viewer, step_delay)

                place_pos     = bin_c.copy()
                place_pos[2] += PLACE_HEIGHT
                _ik_move_carrying(env, dof_addr, place_pos, active_cls,
                                  grasp_offset, viewer, step_delay)

                _open_gripper(env, viewer, step_delay=step_delay)

                obj_pos = env.data.xpos[env._obj_ids[active_cls]].copy()
                placed  = _placed_correctly(obj_pos, active_cls)
                if placed:
                    place_total += 1

            status   = "placed" if placed else ("grasped_no_bin" if (grasped and active_cls not in BIN_CENTERS) else ("grasped" if grasped else "fail"))
            obj_name = CLASS_NAMES.get(active_cls, f"class_{active_cls}") if active_cls is not None else "unknown"
            print(f"episode {ep+1}/{episodes} object={obj_name} reward={total_rew:.2f} steps={n_steps} status={status}")

    env.close()

    n = episodes
    print(f"grasp_success={grasp_total}/{n} ({100*grasp_total/n:.0f}%)")
    if grasp_total:
        print(f"place_success={place_total}/{grasp_total} ({100*place_total/grasp_total:.0f}%)")


if __name__ == "__main__":
    main()
