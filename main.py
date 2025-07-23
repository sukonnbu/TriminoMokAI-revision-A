import os
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from mcts import TriminoMok, Mcts
from cnn import PolicyNetwork, ValueNetwork


def main():
    # Hyperparameters
    num_episodes = 10000
    num_mcts_iterations = 100
    learning_rate = 0.001
    max_depth = 30

    # Initialize networks
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    policy_loaded = False
    value_loaded = False

    policy_episode = 0
    value_episode = 0

    for i in range(num_episodes, 0, -1):
        if (not policy_loaded) and os.path.exists(f"policy_net_{i}.pth"):
            policy_net.load_state_dict(torch.load(f'policy_net_{i}.pth'))
            print("Policy network loaded from file.")
            policy_loaded = True

            policy_episode = i

        if (not value_loaded) and os.path.exists(f"value_net_{i}.pth"):
            value_net.load_state_dict(torch.load(f'value_net_{i}.pth'))
            print("Value network loaded from file.")
            value_loaded = True

            value_episode = i

        if policy_loaded and value_loaded:
            break

    if not policy_loaded:
        # Weight initialization
        for m in policy_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("Policy network initialized.")

    if not value_loaded:
        # Weight initialization
        for m in value_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("Value network initialized.")

    episode = min(policy_episode, value_episode)

    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

    while episode < num_episodes:
        print(f"Episode {episode + 1}/{num_episodes}")

        # Initialize game state
        board, stone_type = init_board()
        game_state = TriminoMok(board, stone_type)
        mcts = Mcts(policy_net, value_net)

        game_history = []

        while not game_state.is_terminal(max_depth):
            # Run MCTS to get the best move
            best_move = mcts.run(game_state, num_mcts_iterations)

            # Store the state and the move probabilities
            game_history.append((game_state.copy_state(), best_move))

            # Make the move
            game_state.make_move(best_move)

        # Determine the winner
        winner = game_state.is_win()

        # Train the networks
        for state_copy, move in game_history:
            board_tensor_policy = torch.from_numpy(TriminoMok(*state_copy).get_board_tensor('policy')).unsqueeze(0).float()
            board_tensor_value = torch.from_numpy(TriminoMok(*state_copy).get_board_tensor('value')).unsqueeze(0).float()

            # Policy network training
            policy_optimizer.zero_grad()
            policy_pred = torch.sigmoid(policy_net(board_tensor_policy))
            policy_target = torch.zeros_like(policy_pred)
            policy_target[0, move[0], move[1], move[2]] = 1.0
            policy_loss = torch.nn.functional.cross_entropy(policy_pred, policy_target)
            policy_loss.backward()
            policy_optimizer.step()

            # Value network training
            value_optimizer.zero_grad()
            value_pred = torch.tanh(value_net(board_tensor_value))
            value_target = torch.tensor([[winner]], dtype=torch.float32)
            value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
            value_loss.backward()
            value_optimizer.step()

        # Save the models
        torch.save(policy_net.state_dict(), f'policy_net_{episode + 1}.pth')
        torch.save(value_net.state_dict(), f'value_net_{episode + 1}.pth')
        
        if os.path.exists(f'policy_net_{episode}.pth'):
            os.remove(f"policy_net_{episode}.pth")
        if os.path.exists(f"value_net_{episode}.pth"):
            os.remove(f"value_net_{episode}.pth")
        
        if episode % 10 == 9:
            torch.save(policy_net.state_dict(), f'saves/policy_net_{episode + 1}.pth')
            torch.save(value_net.state_dict(), f'saves/value_net_{episode + 1}.pth')
        print("Models saved.")

        episode += 1

    print("Training complete and models saved.")


def init_board():
    board = np.zeros((19, 19), dtype=int)

    initial_move = np.random.randint(0, 7)
    center_x = np.random.randint(6, 10)
    center_y = np.random.randint(6, 10)

    board[center_y, center_x] = 1

    match initial_move:
        case 0:
            board[center_y, center_x + 1] = 1
        case 1:
            board[center_y + 1, center_x] = 1
        case 2:
            board[center_y, center_x - 1] = 1
        case 3:
            board[center_y - 1, center_x] = 1
        case 4:
            board[center_y + 1, center_x + 1] = 1
        case 5:
            board[center_y + 1, center_x - 1] = 1
        case 6:
            board[center_y - 1, center_x - 1] = 1
        case 7:
            board[center_y - 1, center_x + 1] = 1

    stone_type = np.random.randint(1, 4)

    return board, stone_type


if __name__ == "__main__":
    main()
