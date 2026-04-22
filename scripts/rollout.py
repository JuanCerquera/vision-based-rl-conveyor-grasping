import os
import sys
import time

import mujoco
import mujoco.viewer

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

def _default_model(algo: str, stage: int) -> str:
    return os.path.join(PROJECT_ROOT, "models", algo, f"stage{stage}", "best", "best_model.zip")


def main():
    algo = "ppo"
    stage = 2
    episodes = 5
    step_delay = 0.04
    deterministic = True

    model_path = _default_model(algo, stage)
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        sys.exit(1)

    from stable_baselines3 import PPO, SAC
    from envs.task1_pick_env import PickEnv

    algo_cls = {"ppo": PPO, "sac": SAC}[algo]
    print(f"Loading {algo.upper()} checkpoint: {model_path}")
    policy = algo_cls.load(model_path)

    stage2 = (stage == 2)
    env = PickEnv(
        active_object   = None if stage2 else 1,
        conveyor_speed  = 0.05,
        randomize_speed = stage2,
        object_classes  = [1, 2] if stage2 else None,
    )

    print(f"Running {episodes} episodes. Close viewer to stop.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        for ep in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
            done = False

            ep_success = False
            min_dist = float('inf')
            while viewer.is_running() and not done:
                action, _ = policy.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated
                min_dist = min(min_dist, env._prev_dist)
                if info.get("success", False):
                    ep_success = True

                viewer.sync()
                time.sleep(step_delay)

            if not viewer.is_running():
                break

            status = "success" if ep_success else "fail"
            print(f"episode {ep+1}/{episodes} steps={step} reward={total_reward:.2f} status={status}")
            print(f"min_dist={min_dist:.4f} termination={info}")
    env.close()


if __name__ == "__main__":
    main()
