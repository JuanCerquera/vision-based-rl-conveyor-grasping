import os
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

STAGE = 1
TIMESTEPS = 5_000_000


def _best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _stage_dir(stage: int) -> str:
    return os.path.join(PROJECT_ROOT, "models", "sac", f"stage{stage}")


def _logs_dir(stage: int) -> str:
    return os.path.join(PROJECT_ROOT, "logs", "sac", f"stage{stage}")


_STAGE_ENV_KWARGS = {
    1: dict(active_object=1,    conveyor_speed=0.05, randomize_speed=False,
            object_classes=None, use_gt_pos=True),
    2: dict(active_object=None, conveyor_speed=0.05, randomize_speed=True,
            object_classes=[1, 2], use_gt_pos=True),
}


def train(stage: int, timesteps: int, device: str):
    from envs.task1_pick_env import PickEnv

    MODELS_DIR = _stage_dir(stage)
    LOGS_DIR   = _logs_dir(stage)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    env_kwargs = _STAGE_ENV_KWARGS[stage]
    env      = make_vec_env(PickEnv, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
    eval_env = make_vec_env(PickEnv, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=MODELS_DIR,
        name_prefix="pick_sac",
        save_replay_buffer=False,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, "best"),
        log_path=LOGS_DIR,
        eval_freq=500_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.02,
        gamma=0.95,
        train_freq=4,
        gradient_steps=4,
        ent_coef=0.1,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        tensorboard_log=LOGS_DIR,
        device=device,
    )

    print(f"Training SAC Stage {stage} on device={device}")
    print(f"Checkpoints: {MODELS_DIR}")
    print(f"TensorBoard: tensorboard --logdir {LOGS_DIR}")

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=True,
    )

    save_path = os.path.join(MODELS_DIR, f"pick_sac_stage{stage}_final")
    model.save(save_path)
    print(f"Stage {stage} complete. Model saved to {save_path}.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    train(STAGE, TIMESTEPS, _best_device())
