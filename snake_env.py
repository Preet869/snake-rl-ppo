"""
Snake Environment — Gymnasium (Gym) format
A pure Python Snake simulator with 10x10 grid.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Constants ---
GRID_SIZE = 10

# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# We use (row, col) for coordinates; row increases downward, col increases rightward
ACTION_DELTAS = {
    0: (-1, 0),  # UP: decrease row
    1: (0, 1),   # RIGHT: increase col
    2: (1, 0),   # DOWN: increase row
    3: (0, -1),  # LEFT: decrease col
}

# Reward shaping
REWARD_FOOD = 1.0
REWARD_DEATH = -1.0
REWARD_STEP = -0.01  # step penalty — encourages efficiency


class SnakeEnv(gym.Env):
    """
    Snake game as a Gymnasium environment.
    Grid is GRID_SIZE x GRID_SIZE. Snake and food are coordinates (row, col).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.grid_size = GRID_SIZE
        self.render_mode = render_mode

        # Action space: 4 discrete actions (up, right, down, left)
        self.action_space = spaces.Discrete(4)

        # Observation: grid matrix (channels-first for SB3 CNN)
        # Values: 0=empty, 1=snake body, 2=snake head, 3=food
        # Shape (1, H, W) so SB3 treats it as a single-channel image
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(1, self.grid_size, self.grid_size),
            dtype=np.uint8,
        )

        # Game state (set in reset)
        self.snake: list[tuple[int, int]] = []  # list of (row, col), [0] = head
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0

    def reset(self, seed=None, options=None):
        """Start a new game."""
        super().reset(seed=seed)

        # Snake starts in the center, length 2 (head + tail)
        center = self.grid_size // 2
        self.snake = [(center, center), (center - 1, center)]  # head at center, tail above
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0

        obs = self._get_obs()
        info = {"score": self.score}
        return obs, info

    def step(self, action):
        """
        Execute one timestep.
        1. Move snake head in direction of action
        2. Check wall collision → game over
        3. Check body collision → game over
        4. If head reaches food → grow, spawn new food, add score
        5. Else → just move (tail shrinks)
        """
        self.steps += 1

        # 1. Compute new head position
        head_row, head_col = self.snake[0]
        dr, dc = ACTION_DELTAS[action]
        new_head = (head_row + dr, head_col + dc)

        # 2. Wall collision?
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._done(obs=self._get_obs(), reward=REWARD_DEATH, truncated=False)

        # 3. Body collision? (new head hits any body segment)
        if new_head in self.snake[:-1]:  # exclude current tail (it will move)
            return self._done(obs=self._get_obs(), reward=REWARD_DEATH, truncated=False)

        # 4. Food?
        if new_head == self.food:
            self.snake = [new_head] + self.snake  # grow: add new head, keep body
            self.score += 1
            self.food = self._spawn_food()
            obs = self._get_obs()
            return obs, REWARD_FOOD + REWARD_STEP, False, False, {"score": self.score}

        # 5. Normal move: add new head, remove tail
        self.snake = [new_head] + self.snake[:-1]

        obs = self._get_obs()
        return obs, REWARD_STEP, False, False, {"score": self.score}

    def _spawn_food(self):
        """Place food on an empty cell (not on snake)."""
        empty = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in self.snake
        ]
        if not empty:
            return (0, 0)  # Snake filled grid — fallback (rare)
        idx = self.np_random.integers(0, len(empty))
        return empty[idx]

    def _get_obs(self):
        """Build observation as grid matrix: 0=empty, 1=body, 2=head, 3=food."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        for r, c in self.snake[1:]:
            grid[r, c] = 1  # body
        grid[self.snake[0][0], self.snake[0][1]] = 2  # head
        grid[self.food[0], self.food[1]] = 3  # food

        return grid[np.newaxis, ...]  # (1, H, W) for channels-first

    def _done(self, obs, reward, truncated):
        """Helper for game-over returns."""
        return obs, reward, True, truncated, {"score": self.score}

    def render(self):
        """Optional: print ASCII grid to console."""
        grid = [["·" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for r, c in self.snake:
            grid[r][c] = "█"
        grid[self.snake[0][0]][self.snake[0][1]] = "●"  # head
        grid[self.food[0]][self.food[1]] = "◆"

        lines = [" ".join(row) for row in grid]
        print("\n".join(lines))
        print(f"Score: {self.score}\n")


# --- Quick test ---
if __name__ == "__main__":
    env = SnakeEnv()
    obs, info = env.reset(seed=42)
    print("Initial obs shape:", obs.shape)
    env.render()

    # Play a few random steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            print("Game over!")
            break
