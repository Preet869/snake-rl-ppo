# Snake RL (PPO)

Snake game trained with PPO (Proximal Policy Optimization) using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). The policy is trained on **20 million timesteps** of experience.

## Setup

```bash
pip install gymnasium numpy stable-baselines3 pygame tensorboard
```


## Train

```bash
python train.py
```

Configurable in `train.py`: `TOTAL_TIMESTEPS`, `MODEL_PATH`, `EVAL_FREQ`, etc. Default training runs for 20M timesteps and saves to `snake_ppo.zip`.

## Play

Watch the trained agent play:

```bash
python play.py --model snake_ppo.zip --episodes 5 --delay 0.15
```

- `--delay`: seconds between steps (higher = slower animation)
- `--episodes`: number of games to run

## Visualize Neural Weights

Generate heatmaps and histograms of the learned policy weights:

```bash
python plot_weights.py --model snake_ppo.zip --out ./weight_viz
```

## Model

- **Architecture**: MlpPolicy (fully connected)
- **Training data**: 20 million timesteps
- **Environment**: 10×10 grid, 500 max steps per episode
- **Rewards**: Food (+1), death (-1), distance shaping (toward/away from food)
