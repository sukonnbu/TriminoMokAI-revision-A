'''
This script allows watching a game played by the AI against itself.
It uses the functional MCTS and game logic modules.
'''

import time
import torch
from random import randint

# Import from the new functional modules
import functional_triminomok as ft
from functional_main import init_board  # Use the functional version of init_board
from functional_mcts import Mcts
from cnn import PolicyNetwork, ValueNetwork

def print_board(board):
    '''Prints the game board to the console.'''
    # Create a string representation of the board for printing
    board_str = "\n".join([" ".join(map(str, row)) for row in board])
    print(board_str)
    print()

def load_trained_models(num_episodes: int, model_num: int) -> tuple:
    '''Loads a specific version of the trained policy and value networks.'''
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    policy_path = f"./saves/setting_1/policy_net_{model_num}.pth"
    value_path = f"./saves/setting_1/value_net_{model_num}.pth"

    try:
        policy_net.load_state_dict(torch.load(policy_path))
        value_net.load_state_dict(torch.load(value_path))
        print(f"Models loaded successfully. Version: {model_num} / {num_episodes}")
        return policy_net, value_net, True
    except FileNotFoundError:
        print(f"Model files not found at {policy_path} or {value_path}.")
        print("Please run the training script (functional_main.py) first.")
        return None, None, False

def main():
    '''Main function to run the AI self-play game.'''
    # Hyperparameters
    num_episodes = 10000
    num_mcts_iterations = 50
    max_depth = 40
    min_depth = 10
    model_to_load = 130  # Developer setting

    # Load models
    policy_net, value_net, models_loaded = load_trained_models(num_episodes, model_to_load)
    if not models_loaded:
        return

    policy_net.eval()  # Set networks to evaluation mode
    value_net.eval()

    # Initialize MCTS with the loaded networks
    mcts = Mcts(policy_net, value_net)

    # Initialize game state using the functional approach
    game_state = init_board()

    print("Starting new game.")
    print("\nTurn 1\nPlayer 1's turn")
    print_board(game_state['board'])

    turn = 1
    play_depth = randint(min_depth, max_depth)

    # Main game loop using functional state transitions
    while not ft.is_terminal(game_state, play_depth):
        turn += 1
        player = game_state['current_player']
        stone_type = game_state['stone_type']
        print(f"Turn {turn}\nPlayer {player}'s turn (Stone type: {stone_type})")

        # Run MCTS to get the best move for the current state
        best_move = mcts.run(game_state, num_mcts_iterations, play_depth)

        if best_move == (0, 0, 0):  # No valid moves available
            print("No valid moves available. Game over.")
            break

        # Get the new game state by applying the move
        game_state = ft.make_move(game_state, best_move)

        # Print the board from the new state
        print(f"Move made: {best_move}")
        print_board(game_state['board'])
        time.sleep(1)  # Pause for readability

    # Determine and announce the winner by calculating the final score
    winner_score = ft.calculate_win_score(game_state)
    print("Game over!")
    if winner_score == 1.0:
        print("Player 2 (White) wins!")
    elif winner_score == -1.0:
        print("Player 1 (Black) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
