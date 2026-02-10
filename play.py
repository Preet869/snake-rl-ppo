"""
Play Snake using the trained PPO policy.

Run (after training has produced snake_ppo.zip):
  python play.py --model snake_ppo.zip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from snake_env import SnakeEnv
from stable_baselines3 import PPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="snake_ppo.zip",
        help="Path to trained PPO model zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds to sleep between steps (slower = more visible)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode before truncation",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found at: {model_path.resolve()}")

    print(f"Loading model from {model_path} ...")
    model = PPO.load(str(model_path))

    env = SnakeEnv(grid_size=10, max_steps=args.max_steps, render_mode="human")

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0

            print(f"\n=== Episode {ep} ===")
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(int(action))
                total_reward += float(reward)

                env.render()
                if args.delay > 0:
                    time.sleep(args.delay)

            print(
                f"Episode {ep} finished "
                f"(terminated={done}, truncated={truncated}), "
                f"score={info.get('score')}, total_reward={total_reward:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

