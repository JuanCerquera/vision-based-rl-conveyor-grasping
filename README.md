# Vision-Based RL Conveyor Grasping

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run full grasp-and-place deployment pipeline
python scripts/deploy.py
# Run policy rollout in the MuJoCo viewer
python scripts/rollout.py
# Evaluate the default trained policy checkpoint
python scripts/evaluate.py
# Generate synthetic data and train YOLO detector
python scripts/train_perception.py
# Train PPO policy with default project settings
python scripts/train_ppo.py
# Train SAC policy with default project settings
python scripts/train_sac.py
```
