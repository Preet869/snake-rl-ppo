"""
Train a PPO policy on Snake.
Run: python train.py
"""

import torch
from snake_env import SnakeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- Config ---
TOTAL_TIMESTEPS = 20_000_000
MODEL_PATH = "snake_ppo.zip"
TB_DIR = "./tb_snake"
EVAL_FREQ = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_env(render_mode=None):
    return Monitor(SnakeEnv(grid_size=10, max_steps=500, render_mode=render_mode))

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    env = DummyVecEnv([lambda: make_env(render_mode=None)])
    eval_env = DummyVecEnv([lambda: make_env(render_mode=None)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device=DEVICE,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=TB_DIR,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,          # <- faster eval
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    env.close()
    eval_env.close()
