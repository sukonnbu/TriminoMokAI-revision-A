import time
import torch
import os.path
from random import randint
from main import init_board
from mcts import TriminoMok, Mcts
from cnn import PolicyNetwork, ValueNetwork


def print_board(board):
    for row in board:
        print(" ".join(map(str, row)))
    print() 


def main():
    # Hyperparameters
    num_episodes = 10000
    num_mcts_iterations = 100
    max_depth = 30
    variation = 5 # 30 +- 5

    # Load trained models
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    model_loaded = False

    for i in range(num_episodes, 0, -1):
        if os.path.exists(f"policy_net_{i}.pth") and os.path.exists(f"value_net_{i}.pth"):
            policy_net.load_state_dict(torch.load(f"policy_net_{i}.pth"))
            value_net.load_state_dict(torch.load(f"value_net_{i}.pth"))
            model_loaded = True
            print(f"Models loaded successfully. Version: {i} / {num_episodes}")
            break

    if not model_loaded:
        print("Model files not found. Please run the training script (main.py) first.")
        return

    policy_net.eval() # Set networks to evaluation mode
    value_net.eval()

    # Initialize game state
    board, stone_type = init_board()
    game_state = TriminoMok(board, stone_type, depth=1)

    print("Starting new game.")
    print("Turn 1\nPlayer 1's turn")
    print_board(game_state.get_board())

    mcts = Mcts(policy_net, value_net)

    turn = 1

    play_depth = randint(max_depth - variation, max_depth + variation)
    while not game_state.is_terminal(play_depth):
        turn += 1
        print(f"Turn {turn}\nPlayer {game_state.get_player()}'s turn (Stone type: {game_state.get_stone_type()})")

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
    elif winner == -1.0:
        print("Player 1 (Black) wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()
