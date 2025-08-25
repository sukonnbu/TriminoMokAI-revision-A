'''
This module contains the main training loop for the TriminoMok AI.
It follows a functional approach, using the state-passing style from 
`functional_triminomok` and `functional_mcts`.

The main loop is broken down into smaller functions for clarity:
- `play_episode`: Simulates a single game and returns the history.
- `train_on_history`: Trains the networks based on the game history.
- `load_models`/`save_models`: Handles model persistence.
'''

import os
import torch
import random
import numpy as np
from torch import nn, optim
from typing import List, Tuple, Dict, Any

# Import from the functional modules
import functional_triminomok as ft
from functional_mcts import Mcts
from cnn import PolicyNetwork, ValueNetwork

GameState = Dict[str, Any]

# --- Hyperparameters ---
NUM_EPISODES = 10000
NUM_MCTS_ITERATIONS = 50
LEARNING_RATE = 0.001
MIN_DEPTH = 10
MAX_DEPTH = 40

def init_board() -> GameState:
    '''Initializes a new game state with a random starting position.'''
    board = np.zeros((19, 19), dtype=int)
    center_x, center_y = np.random.randint(6, 10), np.random.randint(6, 10)
    
    initial_stones = [(center_y, center_x)]
    board[center_y, center_x] = 1

    # A more concise way to handle initial move
    moves = [
        (0, 1), (1, 0), (0, -1), (-1, 0), 
        (1, 1), (1, -1), (-1, -1), (-1, 1)
    ]
    move = random.choice(moves)
    dy, dx = move
    board[center_y + dy, center_x + dx] = 1
    initial_stones.append((center_y + dy, center_x + dx))

    stone_type = np.random.randint(1, 4)
    
    return ft.create_game_state(board, stone_type, prev_actions=[initial_stones], depth=1)

def play_episode(mcts: Mcts) -> Tuple[List[Tuple[GameState, Tuple[int, int, int]]], float]:
    '''Plays one full game episode and returns the history and winner.'''
    game_history = []
    game_state = init_board()
    episode_depth = np.random.randint(MIN_DEPTH, MAX_DEPTH)

    while not ft.is_terminal(game_state, episode_depth):
        best_move = mcts.run(game_state, NUM_MCTS_ITERATIONS, episode_depth)
        if best_move == (0, 0, 0): # Error case or no moves
            break

        game_history.append((game_state, best_move))
        game_state = ft.make_move(game_state, best_move)

    winner = ft.calculate_win_score(game_state)
    return game_history, winner

def train_on_history(
    policy_net: PolicyNetwork, value_net: ValueNetwork, 
    policy_optimizer: optim.Optimizer, value_optimizer: optim.Optimizer, 
    game_history: List[Tuple[GameState, Tuple[int, int, int]]], winner: float
) -> None:
    '''Trains the policy and value networks based on the game history.'''
    for state, move in game_history:
        board_tensor_policy = torch.from_numpy(ft.get_board_tensor(state, 'policy')).unsqueeze(0).float()
        board_tensor_value = torch.from_numpy(ft.get_board_tensor(state, 'value')).unsqueeze(0).float()

        # Policy network training
        policy_optimizer.zero_grad()
        policy_pred = policy_net(board_tensor_policy)
        policy_target = torch.zeros_like(policy_pred.view(-1))
        move_index = move[0] * 19 * 4 + move[1] * 4 + move[2]
        policy_target[move_index] = 1.0
        policy_loss = nn.functional.cross_entropy(policy_pred, policy_target.view(1, 19, 19, 4))
        policy_loss.backward()
        policy_optimizer.step()

        # Value network training
        value_optimizer.zero_grad()
        value_pred = value_net(board_tensor_value)
        value_target = torch.tensor([[winner]], dtype=torch.float32)
        value_loss = nn.functional.mse_loss(value_pred, value_target)
        value_loss.backward()
        value_optimizer.step()

def load_models() -> Tuple[PolicyNetwork, ValueNetwork, int]:
    '''Loads models from disk if they exist, otherwise initializes them.'''
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    start_episode = 0

    # Try to find the latest saved model
    for i in range(NUM_EPISODES, 0, -1):
        policy_path = f'policy_net_{i}.pth'
        value_path = f'value_net_{i}.pth'
        if os.path.exists(policy_path) and os.path.exists(value_path):
            policy_net.load_state_dict(torch.load(policy_path))
            value_net.load_state_dict(torch.load(value_path))
            print(f"Models loaded from episode {i}.")
            start_episode = i
            return policy_net, value_net, start_episode

    # If no models found, initialize weights
    print("Initializing new models.")
    for m in policy_net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.xavier_uniform_(m.weight)
    for m in value_net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.xavier_uniform_(m.weight)
    
    return policy_net, value_net, start_episode

def save_models(policy_net: PolicyNetwork, value_net: ValueNetwork, episode: int) -> None:
    '''Saves models and handles backup copies.'''
    current_policy_path = f'policy_net_{episode}.pth'
    current_value_path = f'value_net_{episode}.pth'
    torch.save(policy_net.state_dict(), current_policy_path)
    torch.save(value_net.state_dict(), current_value_path)

    # Clean up previous model to save space
    prev_policy_path = f'policy_net_{episode - 1}.pth'
    prev_value_path = f'value_net_{episode - 1}.pth'
    if os.path.exists(prev_policy_path): os.remove(prev_policy_path)
    if os.path.exists(prev_value_path): os.remove(prev_value_path)

    # Save a backup every 10 episodes
    if (episode - 1) % 10 == 9:
        backup_policy_path = f'saves/policy_net_{episode}.pth'
        backup_value_path = f'saves/value_net_{episode}.pth'
        torch.save(policy_net.state_dict(), backup_policy_path)
        torch.save(value_net.state_dict(), backup_value_path)
        print(f"Backup models saved for episode {episode}.")


def main():
    '''The main training loop.'''
    policy_net, value_net, start_episode = load_models()
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    value_optimizer = optim.Adam(value_net.parameters(), lr=LEARNING_RATE)
    
    mcts = Mcts(policy_net, value_net)

    for episode in range(start_episode, NUM_EPISODES):
        print(f"Episode {episode + 1}/{NUM_EPISODES}")

        game_history, winner = play_episode(mcts)
        
        print(f"Game finished. Winner: {'Black' if winner == -1.0 else 'White' if winner == 1.0 else 'Draw'}")

        train_on_history(policy_net, value_net, policy_optimizer, value_optimizer, game_history, winner)
        
        save_models(policy_net, value_net, episode + 1)
        print("Models trained and saved.")

    print("Training complete.")

if __name__ == "__main__":
    main()
