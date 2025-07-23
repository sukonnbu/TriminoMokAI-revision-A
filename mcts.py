import torch
import random
from typing import Tuple
from copy import deepcopy
from triminomok import TriminoMok
from cnn import PolicyNetwork, ValueNetwork

class Node:
    def __init__(self, move: Tuple[int, int, int], parent: "Node", state: TriminoMok) -> None:
        self.move = move
        self.parent = parent
        self.state = state
        self.children = []  # A List of Nodes
        self.untried_moves = state.get_moves()  # A List of (y, x, rotation)
        self.visits = 0
        self.value = 0.0

    def select_child(self, policy_values: torch.Tensor):
        children = sorted(self.children, key=lambda c: policy_values[*c.move].item(), reverse=True)

        while True:
            if len(children):
                if children[0].move in self.untried_moves:
                    return children[0]
                else:
                    children.remove(children[0])
            else:
                return self.children[0] # 땜빵

    def add_child(self, move: Tuple[int, int, int], state: TriminoMok):
        child = Node(move, self, state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.visits += 1
        self.value += result


class Mcts:
    def __init__(self, policy: PolicyNetwork, value: ValueNetwork) -> None:
        self._policy_network = policy
        self._value_network = value
        self._max_depth = 0

    def run(self, root_state: TriminoMok, iterations: int, max_depth: int = 30) -> Tuple[int, int, int]:
        self._max_depth = max_depth
        root = Node(None, None, deepcopy(root_state))

        for _ in range(iterations):
            node = self._tree_policy(root)
            self._back_propagate(node)

        if len(root.children) == 0: return 0, 0, 0  # 오류 감지
        return sorted(root.children, key=lambda c: c.value / c.visits, reverse=True)[0].move

    def _tree_policy(self, node: "Node") -> "Node":
        while not node.state.is_terminal(self._max_depth):
            if len(node.untried_moves) != 0:
                return self._expand(node)
            else:
                board_tensor = torch.from_numpy(node.state.get_board_tensor('policy')).unsqueeze(0).float()
                policy_values = self._policy_network(board_tensor)
                node = node.select_child(policy_values)
        return node

    def _expand(self, node: "Node") -> "Node":
        move = random.choice(node.untried_moves)
        next_node = deepcopy(node)
        next_node.state.make_move(move)

        return node.add_child(move, next_node.state)

    def _back_propagate(self, node: "Node") -> None:
        board_tensor = torch.from_numpy(node.state.get_board_tensor('value')).unsqueeze(0).float()
        win_prob = self._value_network(board_tensor).item()

        while not node is None:
            node.update(win_prob)
            win_prob *= 0.05  # 당장을 위해 행동하도록
            node = node.parent
