import os
import sys

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

PPO_HYPERPARAMS = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 256,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    policy_kwargs  = {"net_arch": [64, 64]},
)

STAGE = 1
TIMESTEPS = 24_000_000


def _best_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _stage_dir(stage: int):
    return os.path.join(PROJECT_ROOT, "models", "ppo", f"stage{stage}")


def _logs_dir(stage: int):
    return os.path.join(PROJECT_ROOT, "logs", "ppo", f"stage{stage}")


def train(stage: int, timesteps: int, device: str):
    MODELS_DIR = _stage_dir(stage)
    LOGS_DIR   = _logs_dir(stage)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    from envs.task1_pick_env import PickEnv

    env_kwargs = dict(
        use_gt_pos      = True,
        active_object   = 1 if stage == 1 else None,
        conveyor_speed  = 0.05,
        randomize_speed = (stage == 2),
        object_classes  = None if stage == 1 else [1, 2],
    )

    env      = make_vec_env(PickEnv, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
    eval_env = make_vec_env(PickEnv, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = MODELS_DIR,
        name_prefix = "pick_ppo",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODELS_DIR, "best"),
        log_path             = LOGS_DIR,
        eval_freq            = 500_000,
        n_eval_episodes      = 20,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        **PPO_HYPERPARAMS,
        verbose          = 1,
        tensorboard_log  = LOGS_DIR,
        device           = device,
    )

    print(f"Training PPO Stage {stage} on device={device}")
    print(f"Checkpoints: {MODELS_DIR}")
    print(f"TensorBoard: tensorboard --logdir {LOGS_DIR}")

    model.learn(
        total_timesteps     = timesteps,
        callback            = [checkpoint_cb, eval_cb],
        progress_bar        = True,
        reset_num_timesteps = True,
    )

    save_path = os.path.join(MODELS_DIR, f"pick_ppo_stage{stage}_final")
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    stage = STAGE
    timesteps = TIMESTEPS
    device = _best_device()
    train(stage, timesteps, device)
