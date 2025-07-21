import copy
from copy import deepcopy

import torch
import random
import numpy as np
from typing import List, Tuple
from cnn import PolicyNetwork, ValueNetwork


class TriminoMok:
    def __init__(self, board: np.ndarray, type: int, prev_actions:List[List[Tuple[int, int]]], depth: int = 0, current_player: int = 2) -> None:
        self._current_player = current_player
        self._board = copy.deepcopy(board)
        self._stone_type = type
        self._depth = depth
        self._available_spaces = self.get_available_spaces() # (int, int)
        self._prev_actions = prev_actions  # (y, x)[3][3]
        self._black_score = 0
        self._white_score = 0

    # --- Core Public Methods (for game simulation) ---

    def is_terminal(self, max_depth: int) -> bool:
        return self._depth > max_depth or len(self.get_moves()) == 0

    def get_moves(self) -> List[Tuple[int, int, int]]:
        # (y, x, rotation)
        moves = []

        if self._stone_type == 2:
            for i, j in self._available_spaces:
                stones = self.get_stones(i, j, 0)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 0))

                stones = self.get_stones(i, j, 1)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 1))

                stones = self.get_stones(i, j, 2)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 2))

                stones = self.get_stones(i, j, 3)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 3))
        else:
            for i, j in self._available_spaces:
                stones = self.get_stones(i, j, 0)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 0))

                stones = self.get_stones(i, j, 1)
                if self.is_exists(stones[1]) and self.is_exists(stones[2]):
                    moves.append((i, j, 1))

        return moves

    def make_move(self, move: (int, int, int), add_depth: bool = True) -> None:
        current_stones = self.get_stones(*move)
        if self._prev_actions is None:
            self._prev_actions = []
        elif len(self._prev_actions) == 3:
            self._prev_actions.pop(0)

        self._prev_actions.append(current_stones)

        for i, j in current_stones:
            if random.randint(1, 15) == 1:
                self._board[i][j] = 3
            else:
                self._board[i][j] = self._current_player

        self._black_score += self.get_clear_line(1)
        self._white_score += self.get_clear_line(2)

        self.change_player(add_depth)
        self._available_spaces = self.get_available_spaces()

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
            return -1.0
        else:
            return 0.0

    def copy_state(self):
        return [self._board, self._stone_type, self._prev_actions, self._depth, self._current_player]

    # --- Getters for internal state ---

    def get_board(self):
        return self._board

    def get_player(self):
        return self._current_player

    def get_depth(self):
        return self._depth

    def get_stone_type(self):
        return self._stone_type

    # --- Internal Helper Methods ---

    def change_player(self, add_depth: bool) -> None:
        self._stone_type = random.randint(1, 3)
        if add_depth: self._depth += 1
        self._current_player = 3 - self._current_player

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

    def is_exists(self, pos: Tuple[int, int]) -> bool:
        return pos in self._available_spaces

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

                            checking_stones.append((ni, nj))
                        else:
                            break

                    if count >= 10:
                        clear_lines_count += 1
                        bonus_count += local_bonus_count
                        for y, x in checking_stones:
                            self._board[y, x] = 0

        return clear_lines_count * 20 + bonus_count * 3

    def check_connections(self, i, j, player: int) -> Tuple[float, float, float]:
        connected_lines = 0
        bonus_count = 0
        cleared_lines = 0

        # 가로 탐색
        checking_stones = []
        x_length = 1
        checking_stones.append((i, j))

        # 왼쪽 방향 탐색
        current_x = i
        while True:
            if current_x == 0:
                break
            current_x -= 1
            if self._board[current_x][j] == player:
                x_length += 1
                checking_stones.append((current_x, j))
            elif self._board[current_x][j] == 3:
                x_length += 1
                checking_stones.append((current_x, j))
                bonus_count += 1
            else:
                break

        # 오른쪽 방향 탐색
        current_x = i
        while True:
            if current_x == 18:
                break
            current_x += 1
            if self._board[current_x][j] == player:
                x_length += 1
                checking_stones.append((current_x, j))
            elif self._board[current_x][j] == 3:
                x_length += 1
                checking_stones.append((current_x, j))
                bonus_count += 1
            else:
                break

        if x_length >= 5:
            if x_length >= 10:
                cleared_lines += 1
            else:
                connected_lines += 1

        # 세로 탐색
        checking_stones = []
        y_length = 1
        checking_stones.append((i, j))

        # 위쪽 방향 탐색
        current_y = j
        while True:
            if current_y == 0:
                break
            current_y -= 1
            if self._board[i][current_y] == player:
                y_length += 1
                checking_stones.append((i, current_y))
            elif self._board[i][current_y] == 3:
                y_length += 1
                checking_stones.append((i, current_y))
                bonus_count += 1
            else:
                break

        # 아래쪽 방향 탐색
        current_y = j
        while True:
            if current_y == 18:
                break
            current_y += 1
            if self._board[i][current_y] == player:
                y_length += 1
                checking_stones.append((i, current_y))
            elif self._board[i][current_y] == 3:
                y_length += 1
                checking_stones.append((i, current_y))
                bonus_count += 1
            else:
                break

        if y_length >= 5:
            if y_length >= 10:
                cleared_lines += 1
            else:
                connected_lines += 1

        # x - y = constant 대각선 탐색 (좌상-우하)
        checking_stones = []
        diag_length = 1
        checking_stones.append((i, j))

        # 좌상 방향 탐색
        current_x, current_y = i, j
        while True:
            if current_x == 0 or current_y == 0:
                break
            current_x -= 1
            current_y -= 1
            if self._board[current_x][current_y] == player:
                diag_length += 1
                checking_stones.append((current_x, current_y))
            elif self._board[current_x][current_y] == 3:
                diag_length += 1
                checking_stones.append((current_x, current_y))
                bonus_count += 1
            else:
                break

        # 우하 방향 탐색
        current_x, current_y = i, j
        while True:
            if current_x == 18 or current_y == 18:
                break
            current_x += 1
            current_y += 1
            if self._board[current_x][current_y] == player:
                diag_length += 1
                checking_stones.append((current_x, current_y))
            elif self._board[current_x][current_y] == 3:
                diag_length += 1
                checking_stones.append((current_x, current_y))
                bonus_count += 1
            else:
                break

        if diag_length >= 5:
            if diag_length >= 10:
                cleared_lines += 1
            else:
                connected_lines += 1

        # x + y = constant 대각선 탐색 (우상-좌하)
        checking_stones = []
        diag_length = 1
        checking_stones.append((i, j))

        # 좌하 방향 탐색
        current_x, current_y = i, j
        while True:
            if current_x == 0 or current_y == 18:
                break
            current_x -= 1
            current_y += 1
            if self._board[current_x][current_y] == player:
                diag_length += 1
                checking_stones.append((current_x, current_y))
            elif self._board[current_x][current_y] == 3:
                diag_length += 1
                checking_stones.append((current_x, current_y))
                bonus_count += 1
            else:
                break

        # 우상 방향 탐색
        current_x, current_y = i, j
        while True:
            if current_x == 18 or current_y == 0:
                break
            current_x += 1
            current_y -= 1
            if self._board[current_x][current_y] == player:
                diag_length += 1
                checking_stones.append((current_x, current_y))
            elif self._board[current_x][current_y] == 3:
                diag_length += 1
                checking_stones.append((current_x, current_y))
                bonus_count += 1
            else:
                break

        if diag_length >= 5:
            if diag_length >= 10:
                cleared_lines += 1
            else:
                connected_lines += 1

        return connected_lines, bonus_count, cleared_lines

    # --- AI Interface Method ---

    def get_board_tensor(self, network_type: str):
        if network_type == 'policy':
            board_tensor = np.zeros((14, 19, 19),
                                    dtype=np.float32)  # black, white, bonus, ones, prev*3, conn, bonus_conn, clear, jama, zeros, current_player, stone_shape
        else:  # value
            board_tensor = np.zeros((13, 19, 19), dtype=np.float32)  # black, white, bonus, ones, prev*3, conn, bonus_conn, clear, jama, zeros, current_player

        for times in range(19 * 19):
            i = times // 19
            j = times % 19

            # black, white, bonus
            if self._board[i, j] == 1:
                board_tensor[0, i, j] = 1.0
            elif self._board[i, j] == 2:
                board_tensor[1, i, j] = 1.0
            elif self._board[i, j] == 3:
                board_tensor[2, i, j] = 1.0

            # conn, bonus_conn, clear, jama
            if self._board[i, j] == 0:
                conn, bonus, clear = self.check_connections(i, j, self._current_player)
                board_tensor[7, i, j] = conn
                board_tensor[8, i, j] = bonus
                board_tensor[9, i, j] = clear


        board_tensor[3] = 1.0

        if self._prev_actions is not None:
            player = 3 - self._current_player
            for times in range(len(self._prev_actions)): # 4~6
                for (y, x) in self._prev_actions[times]:
                    board_tensor[3 + times, y, x] = player
                player = 3 - player

        board_tensor[11] = 0.0
        board_tensor[12] = self._current_player

        if network_type == 'policy':
            board_tensor[13] = self._stone_type

        return board_tensor


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
        self.policy_network = policy
        self.value_network = value

    def run(self, root_state: TriminoMok, iterations: int) -> Tuple[int, int, int]:
        root = Node(None, None, copy.deepcopy(root_state))

        for _ in range(iterations):
            node = self._tree_policy(root)
            self._back_propagate(node)

        if len(root.children) == 0: return 0, 0, 0  # 오류 감지
        return sorted(root.children, key=lambda c: c.value / c.visits, reverse=True)[0].move

    def _tree_policy(self, node: "Node") -> "Node":
        while not node.state.is_terminal(10):
            if len(node.untried_moves) != 0:
                return self._expand(node)
            else:
                board_tensor = torch.from_numpy(node.state.get_board_tensor('policy')).unsqueeze(0).float()
                policy_values = self.policy_network(board_tensor)
                node = node.select_child(policy_values)
        return node

    def _expand(self, node: "Node") -> "Node":
        move = random.choice(node.untried_moves)
        next_node = deepcopy(node)
        next_node.state.make_move(move)

        return node.add_child(move, next_node.state)

    def _back_propagate(self, node: "Node") -> None:
        board_tensor = torch.from_numpy(node.state.get_board_tensor('value')).unsqueeze(0).float()
        win_prob = self.value_network(board_tensor).item()

        while not node is None:
            node.update(win_prob)
            win_prob *= 0.1  # 당장을 위해 행동하도록
            node = node.parent


def get_stones(i, j, r, stone_type) -> List[Tuple[int, int]]:
    stones = [(0, 0)] * 3

    if stone_type == 1:
        if r == 0:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i, j + 2)
        else: # r == 1
            stones[0] = (i, j)
            stones[1] = (i + 1, j)
            stones[2] = (i + 2, j)

    elif stone_type == 2:
        if r == 0:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i + 1, j + 1)
        elif r == 1:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i + 1, j)
        elif r == 2:
            stones[0] = (i, j)
            stones[1] = (i, j + 1)
            stones[2] = (i - 1, j + 1)
        else: # r == 3
            stones[0] = (i, j)
            stones[1] = (i + 1, j)
            stones[2] = (i + 1, j + 1)

    else:  # stone_type == 3
        if r == 0:
            stones[0] = (i, j)
            stones[1] = (i - 1, j + 1)
            stones[2] = (i - 2, j + 2)
        else: # r == 1
            stones[0] = (i, j)
            stones[1] = (i + 1, j + 1)
            stones[2] = (i + 2, j + 2)

    return stones
