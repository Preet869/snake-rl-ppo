"""
Plot "neural link" style visuals for a trained PPO Snake policy.

This generates:
- Heatmaps of each linear layer's weight matrix (policy + value nets)
- Weight histograms per layer
- An input-importance map over the grid (derived from the first layer weights)

Run:
  python plot_weights.py --model snake_ppo.zip --out ./weight_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _require_deps():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import torch.nn as nn  # noqa: F401
        from stable_baselines3 import PPO  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies. Make sure you installed: stable-baselines3, torch, matplotlib.\n"
            "Example:\n"
            "  pip install stable-baselines3 torch matplotlib\n"
        ) from e


def _iter_linear_layers(module) -> Iterable[Tuple[str, "np.ndarray"]]:
    import torch.nn as nn

    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy()
            yield name, w


def _save_matrix_heatmap(W: np.ndarray, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    vmax = float(np.max(np.abs(W))) if W.size else 1.0
    plt.imshow(W, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_histogram(W: np.ndarray, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(W.flatten(), bins=80, color="#4C78A8", alpha=0.9)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _infer_square_side(n: int) -> int | None:
    side = int(round(n ** 0.5))
    if side * side == n:
        return side
    return None


def _save_input_importance_map(
    W_first: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """
    Derive an "importance" per input dimension from the first layer weights.
    We use mean(abs(W), axis=0) and reshape if it looks like a square grid.
    """
    import matplotlib.pyplot as plt

    imp = np.mean(np.abs(W_first), axis=0)  # (in_features,)
    side = _infer_square_side(int(imp.shape[0]))

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    if side is not None:
        grid = imp.reshape(side, side)
        plt.imshow(grid, cmap="viridis")
        plt.title(f"{title} ({side}x{side})")
    else:
        plt.plot(imp)
        plt.title(f"{title} (len={imp.shape[0]})")
    plt.colorbar() if side is not None else None
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    _require_deps()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="snake_ppo.zip", help="Path to PPO zip model")
    parser.add_argument("--out", type=str, default="./weight_viz", help="Output directory for images")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise SystemExit(f"Model not found at: {model_path.resolve()}")

    model = PPO.load(str(model_path))
    policy = model.policy

    # Collect linear layers from the shared MLP extractor + heads
    layers = []

    # policy/value MLPs (SB3 ActorCriticPolicy)
    layers.extend([("mlp_extractor", n, w) for n, w in _iter_linear_layers(policy.mlp_extractor)])

    # action head + value head
    try:
        layers.extend([("action_net", n, w) for n, w in _iter_linear_layers(policy.action_net)])
    except Exception:
        pass
    try:
        layers.extend([("value_net", n, w) for n, w in _iter_linear_layers(policy.value_net)])
    except Exception:
        pass

    if not layers:
        raise SystemExit("No linear layers found to plot (unexpected for MlpPolicy).")

    # Save per-layer plots
    for group, name, w in layers:
        safe = (f"{group}_{name}" if name else group).replace(".", "_").replace("/", "_")
        _save_matrix_heatmap(w, f"{group}: {name}  shape={w.shape}", out_dir / f"{safe}_heatmap.png")
        _save_histogram(w, f"{group}: {name} weight histogram", out_dir / f"{safe}_hist.png")

    # Input-importance map from the first MLP layer we can find
    first_layer_w = None
    for group, name, w in layers:
        if w.ndim == 2 and w.shape[1] >= 16:
            first_layer_w = w
            break
    if first_layer_w is not None:
        _save_input_importance_map(
            first_layer_w,
            "Input importance (mean |W| over hidden units)",
            out_dir / "input_importance.png",
        )

    print(f"Saved {len(list(out_dir.glob('*.png')))} images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

