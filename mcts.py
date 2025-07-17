import math
import sys
import numpy as np
import random
import copy
from typing import List, Tuple


class TriminoMok:
    def get_depth(self):
        return self._depth

    def __init__(self, board: np.ndarray, type: int, depth:int=0, current_player:int=2) -> None:
        self._current_player = current_player
        self._board = copy.deepcopy(board)
        self._stone_type = type
        self._depth = depth
        self._available_spaces = []  # (int, int)
        self._black_score = 0
        self._white_score = 0

    def copy_state(self):
        return [self._board, self._stone_type, self._depth, self._current_player]

    def get_board(self):
        return self._board

    def get_stone_type(self):
        return self._stone_type

    def get_available_spaces(self) -> List[Tuple[int, int]]:
        spaces = []
        corners = []

        # 모서리 구하기
        # 왼쪽 방향
        for i in range(19):
            for j in range(1, 19):
                if self._board[i][j - 1] == 0 and self._board[i][j] != 0:
                    corners.append((i, j - 1))

        # 오른쪽 방향
        for i in range(19):
            for j in range(19 - 2, -1, -1):
                if self._board[i][j + 1] == 0 and self._board[i][j] != 0:
                    corners.append((i, j + 1))

        # 위쪽 방향
        for i in range(1, 19):
            for j in range(19):
                if self._board[i - 1][j] == 0 and self._board[i][j] != 0:
                    corners.append((i - 1, j))

        # 아래쪽 방향
        for i in range(19 - 2, -1, -1):
            for j in range(19):
                if self._board[i + 1][j] == 0 and self._board[i][j] != 0:
                    corners.append((i + 1, j))

        # 왼쪽 위 대각선 방향
        for i in range(1, 19):
            for j in range(1, 19):
                if self._board[i - 1][j - 1] == 0 and self._board[i][j] != 0:
                    corners.append((i - 1, j - 1))

        # 오른쪽 아래 대각선 방향
        for i in range(19 - 2, -1, -1):
            for j in range(19 - 2, -1, -1):
                if self._board[i + 1][j + 1] == 0 and self._board[i][j] != 0:
                    corners.append((i + 1, j + 1))

        # 왼쪽 아래 대각선 방향
        for i in range(1, 19):
            for j in range(19 - 2, -1, -1):
                if self._board[i - 1][j + 1] == 0 and self._board[i][j] != 0:
                    corners.append((i - 1, j + 1))

        # 오른쪽 위 대각선 방향
        for i in range(19 - 2, -1, -1):
            for j in range(1, 19):
                if self._board[i + 1][j - 1] == 0 and self._board[i][j] != 0:
                    corners.append((i + 1, j - 1))

        # 공간 더하기
        for i, j in corners:
            for m in range(-2, 3):
                for n in range(-2, 3):
                    if i + m < 0 or i + m > 18 or j + n < 0 or j + n > 18:
                        continue
                    if self._board[i + m][j + n] != 0:
                        continue
                    spaces.append((i + m, j + n))

        return spaces

    def get_stones(self, i, j, r) -> List[Tuple[int, int]]:
        return get_stones(i, j, r, self._stone_type)

    def get_moves(self) -> List[Tuple[int, int, int]]:
        self._available_spaces = self.get_available_spaces()

        # (y, x, rotation)
        moves = []

        if self._stone_type == 2:
            for i, j in self._available_spaces:
                stones = self.get_stones(i, j, 1)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 1))

                stones = self.get_stones(i, j, 2)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 2))

                stones = self.get_stones(i, j, 3)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 3))

                stones = self.get_stones(i, j, 4)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 4))
        else:
            for i, j in self._available_spaces:
                stones = self.get_stones(i, j, 1)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 1))

                stones = self.get_stones(i, j, 2)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 2))

        return moves


    def is_exists(self, pos: Tuple[int, int]) -> bool:
        return pos in self._available_spaces


    def make_move(self, move: (int, int, int), add_depth: bool=True) -> None:
        for i, j in self.get_stones(*move):
            if random.randint(1, 15) == 1:
                self._board[i][j] = 3
            else:
                self._board[i][j] = self._current_player

        self._black_score += self.get_clear_line(1)
        self._white_score += self.get_clear_line(2)

        self.change_player(add_depth)


    def get_clear_line(self, player: int) -> int:
        clear_lines_count = 0
        bonus_count = 0

        # 가로, 세로, 대각선(우하향, 우상향) 방향 벡터
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for i in range(19):
            for j in range(19):
                if self._board[i][j] != player and self._board[i][j] != 3:
                    continue

                # 각 방향에 대해 확인
                for di, dj in directions:
                    checking_stones = [(i, j)]

                    count = 1  # 현재 위치의 돌 포함
                    local_bonus_count = 0

                    # 정방향 확인
                    for step in range(1, 19):
                        ni, nj = i + di * step, j + dj * step

                        if 0 <= ni < 19 and 0 <= nj < 19:
                            if self._board[ni][nj] == player:
                                count += 1
                            elif self._board[ni][nj] == 3:
                                count += 1
                                local_bonus_count += 1
                            else:
                                break

                            checking_stones.append((ni, nj))
                        else:
                            break

                    # 역방향
                    for step in range(1, 19):
                        ni, nj = i - di * step, j - dj * step

                        if 0 <= ni < 19 and 0 <= nj < 19:
                            if self._board[ni][nj] == player:
                                count += 1
                            elif self._board[ni][nj] == 3:
                                count += 1
                                local_bonus_count += 1
                            else:
                                break

                            checking_stones.append((ni,nj))
                        else:
                            break

                    if count >= 10:
                        clear_lines_count += 1
                        bonus_count += local_bonus_count
                        for y, x in checking_stones:
                            self._board[y, x] = 0


        if clear_lines_count: print("Cleared Lines")

        return clear_lines_count * 20 + bonus_count * 3

    def change_player(self, add_depth:bool) -> None:
        self._stone_type = random.randint(1, 3)
        if add_depth: self._depth += 1
        self._current_player = 3 - self._current_player

    def is_win(self) -> float:
        scores = [0, 0]  # 플레이어 1과 2의 점수

        # 가로 연결 탐색
        for y in range(18, -1, -1):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            for x in range(19):
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                    continue

                # 첫 돌이 놓여있을 때
                if checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                # 같은 색 돌이 놓여있을 때
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                # 보너스 돌이 놓여있을 때
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                # 다른 색 돌이 놓여있을 때
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

            # 각 행의 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 세로 연결 탐색
        for x in range(19):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            for y in range(18, -1, -1):
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                    continue

                # 첫 돌이 놓여있을 때
                if checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                # 같은 색 돌이 놓여있을 때
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                # 보너스 돌이 놓여있을 때
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                # 다른 색 돌이 놓여있을 때
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

            # 각 열의 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 왼쪽 위에서 오른쪽 아래 대각선 탐색 (오른쪽으로 이동)
        for start_x in range(19):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            x, y = start_x, 0
            while x < 19 and y < 19:
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                elif checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

                x += 1
                y += 1

            # 대각선 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 왼쪽 위에서 오른쪽 아래 대각선 탐색 (아래로 이동)
        for start_y in range(1, 19):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            x, y = 0, start_y
            while x < 19 and y < 19:
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                elif checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

                x += 1
                y += 1

            # 대각선 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 오른쪽 위에서 왼쪽 아래 대각선 탐색 (왼쪽으로 이동)
        for start_x in range(18, -1, -1):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            x, y = start_x, 0
            while x >= 0 and y < 19:
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                elif checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

                x -= 1
                y += 1

            # 대각선 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 오른쪽 위에서 왼쪽 아래 대각선 탐색 (아래로 이동)
        for start_y in range(1, 19):
            checking_stone_type = 0
            connection_length = 0
            bonus_number = 0
            x, y = 18, start_y
            while x >= 0 and y < 19:
                if self._board[x][y] == 0:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = 0
                    connection_length = 0
                    bonus_number = 0
                elif checking_stone_type == 0:
                    if self._board[x][y] != 3:
                        checking_stone_type = self._board[x][y]
                    connection_length += 1
                elif self._board[x][y] == checking_stone_type:
                    connection_length += 1
                elif self._board[x][y] == 3:
                    connection_length += 1
                    bonus_number += 1
                else:
                    if connection_length >= 5:
                        scores[checking_stone_type - 1] += 3 + bonus_number * 1

                    checking_stone_type = self._board[x][y]
                    connection_length = 1
                    bonus_number = 0

                x -= 1
                y += 1

            # 대각선 끝에서 연결 확인
            if connection_length >= 5 and checking_stone_type > 0:
                scores[checking_stone_type - 1] += 3 + bonus_number * 1

        # 점수 비교하여 승자 결정
        self._black_score += scores[0]
        self._white_score += scores[1]

        if self._white_score > self._black_score:
            return 1.0
        elif self._white_score < self._black_score:
            return 0.0
        else:
            return 0.5

    def is_terminal(self, max_depth: int) -> bool:
        return self._depth > max_depth or len(self.get_moves()) == 0

class Node:
    def __init__(self, move, parent, state: TriminoMok) -> None:
        self.move = move
        self.parent = parent
        self.state = state
        self.children = [] # A List of Nodes
        self.untried_moves = state.get_moves() # A List of (y, x, rotation)
        self.visits = 0
        self.wins = 0

    def ucb1(self, parent_visits: int) -> float:
        if self.visits == 0: return sys.float_info.max
        else: return self.wins / self.visits + 1.4 * math.sqrt(math.log(parent_visits) / self.visits)

    def select_child(self):
        return sorted(self.children, key=lambda c: -c.ucb1(self.visits))[0] # 내림차순 정렬

    def add_child(self, move: Tuple[int, int, int], state: TriminoMok):
        child = Node(move, self, state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result: float) -> None:
        self.visits += 1
        self.wins += result

class Mcts:
    def run(self, root_state: TriminoMok, iterations: int) -> Tuple[int, int, int]:
        root = Node(None, None, copy.deepcopy(root_state))

        for _ in range(iterations):
            node = self._tree_policy(root)
            wins = self._default_policy(node.state)
            self._back_propagate(node, wins)

        if len(root.children) == 0: return 0, 0, 0  # 오류 감지
        return sorted(root.children, key=lambda c: - c.wins / c.visits)[0].move

    def _tree_policy(self, node: "Node") -> "Node":
        while not node.state.is_terminal(5):
            if len(node.untried_moves) != 0: return self._expand(node)
            else: node = node.select_child()
        return node

    def _expand(self, node: "Node") -> "Node":
        move = random.choice(node.untried_moves)
        node.state.make_move(move)

        return node.add_child(move, node.state)

    def _default_policy(self, state: "TriminoMok") -> float:
        current_state = state

        while not current_state.is_terminal(5):
            moves = current_state.get_moves()
            if len(moves) == 0: break
            state.make_move(random.choice(moves))

        return state.is_win()

    def _back_propagate(self, node: "Node", wins: float) -> None:
        while node != None:
            node.update(wins)
            node = node.parent

def get_stones(i, j, r, stone_type) -> List[Tuple[int, int]]:
    stones = [(0, 0)] * 3

    if stone_type == 1:
        if r == 1:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i, j + 2)
        else:
            stones[0] = (i, j)
            stones[1] = (i + 1, j)
            stones[2] = (i + 2, j)

    elif stone_type == 2:
        if r == 1:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i + 1, j + 1)
        elif r == 2:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i + 1, j)
        elif r == 3:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i - 1, j + 1)
        else:
            stones[0] = (i, j)
            stones[1] = (i + 1, j)
            stones[2] = (i + 1, j + 1)

    else:  # stone_type == 3
        if r == 1:
            stones[0] = (i, j)
            stones[1] = (i - 1, j + 1)
            stones[2] = (i - 2, j + 2)
        else:
            stones[0] = (i, j)
            stones[1] = (i + 1, j + 1)
            stones[2] = (i + 2, j + 2)

    return stones