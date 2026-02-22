"""9x9 Four-in-a-Row with a DQN agent.

Features:
- 9x9 board without gravity.
- Human vs computer game mode.
- Self-play training with DQN (experience replay + target network).

Usage examples:
    python connect4_dqn.py train --episodes 3000 --save model.pth
    python connect4_dqn.py play --model model.pth
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


BOARD_SIZE = 9
CONNECT_N = 4


class Connect4Env:
    """9x9 four-in-a-row environment with alternating turns (no gravity)."""

    def __init__(self, size: int = BOARD_SIZE, connect_n: int = CONNECT_N):
        self.size = size
        self.connect_n = connect_n
        self.board = np.zeros((size, size), dtype=np.int8)

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        return self.board.copy()

    def valid_actions(self) -> List[int]:
        return [r * self.size + c for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]

    def step(self, action: int, player: int) -> Tuple[np.ndarray, float, bool]:
        if action not in self.valid_actions():
            return self.board.copy(), -1.0, True

        row, col = divmod(action, self.size)
        self.board[row, col] = player

        if self.check_winner(player):
            return self.board.copy(), 1.0, True

        if not self.valid_actions():
            return self.board.copy(), 0.0, True

        return self.board.copy(), 0.0, False

    def check_winner(self, player: int) -> bool:
        n = self.connect_n
        b = self.board
        s = self.size

        for r in range(s):
            for c in range(s):
                if b[r, c] != player:
                    continue

                if c + n <= s and all(b[r, c + i] == player for i in range(n)):
                    return True
                if r + n <= s and all(b[r + i, c] == player for i in range(n)):
                    return True
                if r + n <= s and c + n <= s and all(b[r + i, c + i] == player for i in range(n)):
                    return True
                if r + n <= s and c - n + 1 >= 0 and all(b[r + i, c - i] == player for i in range(n)):
                    return True

        return False

    def render(self) -> None:
        symbols = {0: ".", 1: "X", -1: "O"}
        print("  " + " ".join(str(c) for c in range(self.size)))
        for r in range(self.size):
            print(f"{r} " + " ".join(symbols[self.board[r, c]] for c in range(self.size)))
        print()


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    valid_next_actions: List[int]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        device: str = "cpu",
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.q_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def encode_state(self, board: np.ndarray, perspective: int) -> np.ndarray:
        # Normalize to perspective of current player: own pieces -> 1, opponent -> -1
        return (board * perspective).astype(np.float32).flatten()

    def select_action(self, state: np.ndarray, valid_actions: List[int], greedy: bool = False) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available")

        if (not greedy) and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()

        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> float:
        if len(replay) < batch_size:
            return 0.0

        batch = replay.sample(batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_net(next_states).cpu().numpy()
            next_q = []
            for i, t in enumerate(batch):
                if t.done or not t.valid_next_actions:
                    next_q.append(0.0)
                else:
                    next_q.append(float(np.max(next_q_all[i, t.valid_next_actions])))
            next_q = torch.tensor(next_q, dtype=torch.float32, device=self.device)
            target = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state)
        self.target_net.load_state_dict(state)


def self_play_train(
    episodes: int = 5000,
    batch_size: int = 128,
    target_update_interval: int = 100,
    model_path: str = "model.pth",
    device: str = "cpu",
) -> None:
    env = Connect4Env()
    agent = DQNAgent(state_size=BOARD_SIZE * BOARD_SIZE, action_size=BOARD_SIZE * BOARD_SIZE, device=device)
    replay = ReplayBuffer()

    for ep in range(1, episodes + 1):
        env.reset()
        done = False
        current_player = 1
        pending_state = {1: None, -1: None}
        pending_action = {1: None, -1: None}

        while not done:
            valid_actions = env.valid_actions()
            state = agent.encode_state(env.board, current_player)
            action = agent.select_action(state, valid_actions)

            next_board, reward, done = env.step(action, current_player)
            next_state_for_self = agent.encode_state(next_board, current_player)

            pending_state[current_player] = state
            pending_action[current_player] = action

            if done:
                replay.add(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state_for_self,
                        done=True,
                        valid_next_actions=[],
                    )
                )
                other = -current_player
                if pending_state[other] is not None:
                    replay.add(
                        Transition(
                            state=pending_state[other],
                            action=pending_action[other],
                            reward=-reward,
                            next_state=agent.encode_state(next_board, other),
                            done=True,
                            valid_next_actions=[],
                        )
                    )
            else:
                other = -current_player
                if pending_state[other] is not None:
                    replay.add(
                        Transition(
                            state=pending_state[other],
                            action=pending_action[other],
                            reward=0.0,
                            next_state=agent.encode_state(next_board, other),
                            done=False,
                            valid_next_actions=env.valid_actions(),
                        )
                    )

            agent.train_step(replay, batch_size)
            current_player *= -1

        if ep % target_update_interval == 0:
            agent.update_target()

        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} | epsilon={agent.epsilon:.3f} | replay={len(replay)}")

    agent.save(model_path)
    print(f"Training complete. Model saved to: {model_path}")


def play_vs_human(model_path: str, device: str = "cpu") -> None:
    env = Connect4Env()
    agent = DQNAgent(state_size=BOARD_SIZE * BOARD_SIZE, action_size=BOARD_SIZE * BOARD_SIZE, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0

    env.reset()
    human_player = 1
    ai_player = -1
    current_player = human_player

    print("You are X. Enter row and column index (0-8 0-8).")
    env.render()

    while True:
        if current_player == human_player:
            valid = env.valid_actions()
            while True:
                try:
                    row, col = map(int, input("Your move (row col): ").strip().split())
                except ValueError:
                    print("Please input two integers: row col")
                    continue
                action = row * BOARD_SIZE + col
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and action in valid:
                    break
                print("Invalid cell.")

            _, reward, done = env.step(action, human_player)
            env.render()
            if done:
                if reward > 0:
                    print("You win!")
                elif reward == 0:
                    print("Draw.")
                else:
                    print("Invalid move - you lose.")
                return
        else:
            state = agent.encode_state(env.board, ai_player)
            action = agent.select_action(state, env.valid_actions(), greedy=True)
            ai_row, ai_col = divmod(action, BOARD_SIZE)
            print(f"AI move: row={ai_row}, col={ai_col}")
            _, reward, done = env.step(action, ai_player)
            env.render()
            if done:
                if reward > 0:
                    print("AI wins.")
                elif reward == 0:
                    print("Draw.")
                else:
                    print("AI made invalid move (unexpected).")
                return

        current_player *= -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9x9 Four-in-a-Row with DQN self-play")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_p = subparsers.add_parser("train", help="self-play training")
    train_p.add_argument("--episodes", type=int, default=3000)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--target-update", type=int, default=100)
    train_p.add_argument("--save", type=str, default="model.pth")
    train_p.add_argument("--device", type=str, default="cpu")

    play_p = subparsers.add_parser("play", help="play versus trained model")
    play_p.add_argument("--model", type=str, default="model.pth")
    play_p.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        self_play_train(
            episodes=args.episodes,
            batch_size=args.batch_size,
            target_update_interval=args.target_update,
            model_path=args.save,
            device=args.device,
        )
    elif args.mode == "play":
        play_vs_human(args.model, device=args.device)


if __name__ == "__main__":
    main()
