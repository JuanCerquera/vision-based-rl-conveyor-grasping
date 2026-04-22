import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

ENV_REGISTRY = {
    "task1": ("envs.task1_pick_env", "PickEnv"),
}

CLASS_NAMES = ["cereal_box", "soda_can", "milk_carton", "tennis_ball"]
ALGO = "ppo"
STAGE = 2
EPISODES = 20
DETERMINISTIC = True


def _default_model(algo: str, stage: int) -> str:
    return os.path.join(PROJECT_ROOT, "models", algo, f"stage{stage}", "best", "best_model.zip")


_STAGE_ENV_KWARGS = {
    1: dict(active_object=1, conveyor_speed=0.05, randomize_speed=False),
    2: dict(active_object=None, object_classes=[1, 2], conveyor_speed=0.05, randomize_speed=True),
}


def evaluate(model_path: str, algo: str, env_name: str, n_episodes: int, deterministic: bool, stage: int = 1):
    import importlib
    from stable_baselines3 import PPO, SAC

    module_path, class_name = ENV_REGISTRY[env_name]
    EnvCls = getattr(importlib.import_module(module_path), class_name)

    algo_cls = {"ppo": PPO, "sac": SAC}[algo]
    print(f"Loading {algo.upper()} checkpoint: {model_path}")
    policy = algo_cls.load(model_path)

    env_kwargs = _STAGE_ENV_KWARGS.get(stage, _STAGE_ENV_KWARGS[1])
    print(f"Stage {stage} env kwargs: {env_kwargs}")
    env = EnvCls(**env_kwargs)

    episode_rewards  = []
    episode_lengths  = []
    episode_success  = []
    class_results = defaultdict(list)

    print(f"Running {n_episodes} episodes deterministic={deterministic}")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        active_class = env._active_idx

        total_reward = 0.0
        steps = 0
        done = False
        success = False

        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if info.get("success", False):
                success = True

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_success.append(success)
        class_results[active_class].append(success)

        status = "success" if success else "fail"
        print(f"episode {ep+1}/{n_episodes} object={CLASS_NAMES[active_class]} reward={total_reward:.2f} steps={steps} status={status}")

    env.close()

    rewards  = np.array(episode_rewards)
    lengths  = np.array(episode_lengths)
    successes = np.array(episode_success)

    print(f"episodes={n_episodes}")
    print(f"success={successes.sum()}/{n_episodes} ({100*successes.mean():.1f}%)")
    print(f"reward_mean={rewards.mean():.2f} reward_std={rewards.std():.2f} reward_min={rewards.min():.2f} reward_max={rewards.max():.2f}")
    print(f"length_mean={lengths.mean():.1f} length_std={lengths.std():.1f}")
    print("per_class:")
    for cls_id, name in enumerate(CLASS_NAMES):
        results = class_results[cls_id]
        if not results:
            print(f"{name}: no_episodes")
            continue
        rate = 100 * sum(results) / len(results)
        print(f"{name}: {sum(results)}/{len(results)} ({rate:.0f}%)")


if __name__ == "__main__":
    env_name = "task1"
    model_path = _default_model(ALGO, STAGE)
    if not os.path.exists(model_path):
        print(f"Checkpoint not found: {model_path}")
        sys.exit(1)

    evaluate(model_path, ALGO, env_name, EPISODES, deterministic=DETERMINISTIC, stage=STAGE)
