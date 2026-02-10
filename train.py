"""
Train a PPO policy on Snake.
Run: python train.py
"""

import torch
from snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# --- Config ---
TOTAL_TIMESTEPS = 200_000
MODEL_PATH = "snake_ppo.zip"
LOG_DIR = "logs/ppo_snake"
EVAL_FREQ = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    env = DummyVecEnv([lambda: SnakeEnv()])
    eval_env = DummyVecEnv([lambda: SnakeEnv()])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device=DEVICE,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./",
        log_path="./",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    env.close()
    eval_env.close()
