import torch
import numpy as np
import time

from mcts import TriminoMok, Mcts
from cnn import PolicyNetwork, ValueNetwork

def print_board(board):
    for row in board:
        print(" ".join(map(str, row)))
    print() 

def main():
    # Load trained models
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    try:
        policy_net.load_state_dict(torch.load("policy_net.pth"))
        value_net.load_state_dict(torch.load("value_net.pth"))
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Model files not found. Please run the training script (main.py) first.")
        return

    policy_net.eval() # Set networks to evaluation mode
    value_net.eval()

    # Initialize game state
    board = np.zeros((19, 19), dtype=int)
    board[8, 8] = 1
    board[8, 9] = 1
    initial_stone_type = np.random.randint(1, 4)
    game_state = TriminoMok(board, initial_stone_type, prev_actions=None)

    print("Starting new game. Initial board:")
    print_board(game_state.get_board())

    num_mcts_iterations = 50 # Use the same number of iterations as in training for consistency

    mcts = Mcts(policy_net, value_net)

    while not game_state.is_terminal(max_depth=30):
        print(f"Player {game_state.get_player()}'s turn (Stone type: {game_state.get_stone_type()})")

        # Get the best move from the AI
        best_move = mcts.run(game_state, num_mcts_iterations)

        if best_move == (0,0,0):
            print("No valid moves available. Game over.")
            break

        # Make the move
        game_state.make_move(best_move)

        # Print the board
        print(f"Move made: {best_move}")
        print_board(game_state.get_board())
        time.sleep(1) # Pause for a moment to make it easier to follow

    # Determine and announce the winner
    winner = game_state.is_win()
    print("Game over!")
    if winner == 1.0:
        print("Player 2 (White) wins!")
    elif winner == 0.0:
        print("Player 1 (Black) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
