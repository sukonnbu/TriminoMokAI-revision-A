'''
This module implements a Monte Carlo Tree Search (MCTS) algorithm designed to work
with the functional game logic from `functional_triminomok`.

The MCTS nodes store the game state as an immutable dictionary, and the tree is expanded
using the pure functions from the game logic module.
'''

import torch
import random
from typing import Tuple, Dict, Any
from copy import deepcopy

# Import functions from the functional implementation of the game
import functional_triminomok as ft
from cnn import PolicyNetwork, ValueNetwork

# Type alias for game state for clarity
GameState = Dict[str, Any]

class Node:
    '''Represents a node in the MCTS tree.'''
    def __init__(self, move: Tuple[int, int, int], parent: "Node", state: GameState) -> None:
        self.move = move
        self.parent = parent
        self.state = state  # The state is now a dictionary
        self.children = []
        self.untried_moves = ft.get_moves(state)  # Use functional get_moves
        self.visits = 0
        self.value = 0.0

    def select_child(self, policy_values: torch.Tensor) -> "Node":
        '''Selects a child node based on policy values.'''
        # Sort children by their corresponding policy value in descending order
        children_sorted_by_policy = sorted(
            self.children, 
            key=lambda c: policy_values[0, c.move[0] * 19 * 4 + c.move[1] * 4 + c.move[2]].item(), 
            reverse=True
        )

        # Find the best valid child
        for child in children_sorted_by_policy:
            if not ft.is_terminal(child.state, 30): # Assuming max_depth of 30
                return child
        
        # Fallback: if all children lead to terminal states, return the best one anyway
        return children_sorted_by_policy[0] if children_sorted_by_policy else self.children[0]

    def add_child(self, move: Tuple[int, int, int], state: GameState) -> "Node":
        '''Adds a new child node to this node.'''
        child = Node(move, self, state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        '''Updates the node's value and visit count.'''
        self.visits += 1
        self.value += result


class Mcts:
    '''Manages the MCTS process.'''
    def __init__(self, policy: PolicyNetwork, value: ValueNetwork) -> None:
        self._policy_network = policy
        self._value_network = value
        self._max_depth = 0

    def run(self, root_state: GameState, iterations: int, max_depth: int = 30) -> Tuple[int, int, int]:
        '''Runs the MCTS algorithm for a given number of iterations.'''
        self._max_depth = max_depth
        # The root node is initialized with the functional game state
        root = Node(None, None, deepcopy(root_state))

        for _ in range(iterations):
            node = self._tree_policy(root)
            self._back_propagate(node)

        if not root.children:
            return 0, 0, 0  # No possible moves

        # Select the move of the child with the highest average value
        best_child = max(root.children, key=lambda c: c.value / c.visits)
        return best_child.move

    def _tree_policy(self, node: "Node") -> "Node":
        '''Selects or expands a node in the tree.'''
        while not ft.is_terminal(node.state, self._max_depth):
            if len(node.untried_moves) != 0:
                return self._expand(node)
            else:
                if not node.children:
                    # If there are no children and no untried moves, it's a terminal leaf
                    return node
                board_tensor = torch.from_numpy(ft.get_board_tensor(node.state, 'policy')).unsqueeze(0).float()
                policy_values = self._policy_network(board_tensor)
                node = node.select_child(policy_values)
        return node

    def _expand(self, node: "Node") -> "Node":
        '''Expands the tree by adding a new child node.'''
        move = random.choice(node.untried_moves)
        
        # Create the new state by calling the pure make_move function
        next_state = ft.make_move(node.state, move)
        
        return node.add_child(move, next_state)

    def _back_propagate(self, node: "Node") -> None:
        '''Backpropagates the simulation result up the tree.'''
        # Get the win probability from the value network
        board_tensor = torch.from_numpy(ft.get_board_tensor(node.state, 'value')).unsqueeze(0).float()
        win_prob = self._value_network(board_tensor).item()

        # Update nodes from the leaf back to the root
        current_node = node
        while current_node is not None:
            current_node.update(win_prob)
            # The reward is discounted for parent nodes to encourage decisive moves
            win_prob *= 0.05  
            current_node = current_node.parent
