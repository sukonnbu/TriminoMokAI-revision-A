import torch
import numpy as np
from mcts import TriminoMok, Mcts
from cnn import PolicyNetwork, ValueNetwork
from torch import nn
import torch.optim as optim
import os

def main():
    # Hyperparameters
    num_episodes = 1000
    num_mcts_iterations = 50
    learning_rate = 0.001

    # Initialize networks
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()

    if os.path.exists('policy_net.pth'):
        policy_net.load_state_dict(torch.load('policy_net.pth'))
        print("Policy network loaded from file.")
    else:
        # Weight initialization
        for m in policy_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("Policy network initialized.")

    if os.path.exists('value_net.pth'):
        value_net.load_state_dict(torch.load('value_net.pth'))
        print("Value network loaded from file.")
    else:
        # Weight initialization
        for m in value_net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("Value network initialized.")

    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Initialize game state
        board = np.zeros((19, 19), dtype=int)
        board[8,8] = 1
        board[8,9] = 1
        initial_stone_type = np.random.randint(1, 4)
        game_state = TriminoMok(board, initial_stone_type, None)
        mcts = Mcts(policy_net, value_net)

        game_history = []

        while not game_state.is_terminal(max_depth=30):
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
        torch.save(policy_net.state_dict(), 'policy_net.pth')
        torch.save(value_net.state_dict(), 'value_net.pth')
        
        if episode % 50 == 49:
            torch.save(policy_net.state_dict(), f'saves/policy_net_{episode + 1}.pth')
            torch.save(value_net.state_dict(), f'saves/value_net_{episode + 1}.pth')
        print("Models saved.")

    print("Training complete and models saved.")


if __name__ == "__main__":
    main()
