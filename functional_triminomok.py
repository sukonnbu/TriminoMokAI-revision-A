'''
This module implements the TriminoMok game logic using a functional programming paradigm.
Instead of a class, the game state is represented by a dictionary, and functions operate
on this state to produce a new state, avoiding side effects.
'''

import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict, Any

# Type alias for game state for better readability
GameState = Dict[str, Any]


# --- Game State Creation ---

def create_game_state(board: np.ndarray, stone_type: int, prev_actions: List[List[Tuple[int, int]]] = None, depth: int = 0, current_player: int = 2) -> GameState:
    '''Creates a new game state dictionary.'''
    return {
        "current_player": current_player,
        "board": deepcopy(board),
        "stone_type": stone_type,
        "depth": depth,
        "available_spaces": get_available_spaces(board),
        "prev_actions": prev_actions if prev_actions is not None else [],
        "black_score": 0,
        "white_score": 0,
    }

# --- Core Public Functions (Pure Functions) ---

def is_terminal(game_state: GameState, max_depth: int) -> bool:
    '''Checks if the game has reached a terminal state.'''
    return game_state["depth"] >= max_depth or len(get_moves(game_state)) == 0

def get_moves(game_state: GameState) -> List[Tuple[int, int, int]]:
    '''Gets all possible moves for the current player.'''
    moves = []
    available_spaces = game_state["available_spaces"]
    stone_type = game_state["stone_type"]

    def is_valid_placement(stones: List[Tuple[int, int]]) -> bool:
        '''Checks if all stones in a trimino can be placed on available spaces.'''
        return all(pos in available_spaces for pos in stones[1:])

    rotations = range(4) if stone_type == 2 else range(2)

    for i, j in available_spaces:
        for r in rotations:
            stones = get_stones(i, j, r, stone_type)
            if is_valid_placement(stones):
                moves.append((i, j, r))

    return moves

def make_move(game_state: GameState, move: Tuple[int, int, int], add_depth: bool = True) -> GameState:
    '''Applies a move and returns a new game state without modifying the original.'''
    new_board = deepcopy(game_state["board"])
    current_player = game_state["current_player"]

    # Place stones
    current_stones = get_stones(*move, game_state["stone_type"])
    for i, j in current_stones:
        if 0 <= i < 19 and 0 <= j < 19:
            new_board[i, j] = 3 if random.randint(1, 15) == 1 else current_player

    # Clear lines and update scores
    black_clear_score, new_board = get_clear_line(new_board, 1)
    white_clear_score, new_board = get_clear_line(new_board, 2)

    new_black_score = game_state["black_score"] + black_clear_score
    new_white_score = game_state["white_score"] + white_clear_score

    # Update previous actions
    new_prev_actions = game_state["prev_actions"][:]
    if len(new_prev_actions) == 3:
        new_prev_actions.pop(0)
    new_prev_actions.append(current_stones)

    # Create the next state
    next_state = {
        "current_player": 3 - current_player,
        "board": new_board,
        "stone_type": random.randint(1, 3),
        "depth": game_state["depth"] + 1 if add_depth else game_state["depth"],
        "available_spaces": get_available_spaces(new_board),
        "prev_actions": new_prev_actions,
        "black_score": new_black_score,
        "white_score": new_white_score,
    }
    return next_state

def calculate_win_score(game_state: GameState) -> float:
    '''Calculates the final score difference to determine the winner.'''
    board = game_state["board"]
    scores = [0, 0]  # Player 1 (black), Player 2 (white)

    # Helper to process a line of stones
    def process_line(line):
        nonlocal scores
        checking_stone_type = 0
        connection_length = 0
        bonus_number = 0
        for stone in line:
            if stone == 0:
                if connection_length >= 5:
                    scores[checking_stone_type - 1] += 3 + bonus_number
                checking_stone_type, connection_length, bonus_number = 0, 0, 0
                continue

            if checking_stone_type == 0:
                if stone != 3: checking_stone_type = stone
                connection_length = 1
            elif stone == checking_stone_type or stone == 3:
                connection_length += 1
                if stone == 3: bonus_number += 1
            else:
                if connection_length >= 5:
                    scores[checking_stone_type - 1] += 3 + bonus_number
                checking_stone_type = stone if stone != 3 else 0
                connection_length = 1
                bonus_number = 0
        
        if connection_length >= 5 and checking_stone_type > 0:
            scores[checking_stone_type - 1] += 3 + bonus_number

    # Horizontal, Vertical, and Diagonal lines
    for i in range(19):
        process_line(board[:, i])  # Vertical
        process_line(board[i, :])  # Horizontal

    for k in range(-18, 19):
        process_line(np.diag(board, k=k))
        process_line(np.diag(np.fliplr(board), k=k))

    final_black_score = game_state["black_score"] + scores[0]
    final_white_score = game_state["white_score"] + scores[1]

    if final_white_score > final_black_score:
        return 1.0
    elif final_white_score < final_black_score:
        return -1.0
    else:
        return 0.0

# --- Internal Helper Functions ---

def get_available_spaces(board: np.ndarray) -> List[Tuple[int, int]]:
    '''Gets a list of all empty coordinates on the board.'''
    return list(zip(*np.where(board == 0)))

def get_stones(i: int, j: int, r: int, stone_type: int) -> List[Tuple[int, int]]:
    '''Calculates the coordinates of a trimino based on type and rotation.'''
    stones = [(0, 0)] * 3
    if stone_type == 1: # I-shape
        if r == 0: stones = [(i, j), (i, j + 1), (i, j + 2)]
        else:      stones = [(i, j), (i + 1, j), (i + 2, j)]
    elif stone_type == 2: # L-shape
        if r == 0:   stones = [(i, j), (i, j + 1), (i + 1, j + 1)]
        elif r == 1: stones = [(i, j), (i, j + 1), (i + 1, j)]
        elif r == 2: stones = [(i, j), (i, j + 1), (i - 1, j + 1)]
        else:        stones = [(i, j), (i + 1, j), (i + 1, j + 1)]
    else: # Diagonal shape
        if r == 0:   stones = [(i, j), (i - 1, j + 1), (i - 2, j + 2)]
        else:        stones = [(i, j), (i + 1, j + 1), (i + 2, j + 2)]
    return stones

def get_clear_line(board: np.ndarray, player: int) -> Tuple[int, np.ndarray]:
    '''Checks for and clears lines of 10 or more stones, returning the score and new board.'''
    new_board = deepcopy(board)
    clear_lines_count = 0
    bonus_count = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    coords_to_clear = set()

    for r in range(19):
        for c in range(19):
            if board[r, c] != player and board[r, c] != 3:
                continue

            for dr, dc in directions:
                line_coords = []
                local_bonus = 0
                # Check in both positive and negative directions from the current stone
                for sign in [-1, 1]:
                    # Start from the stone itself for the positive direction search
                    start_step = 0 if sign == 1 else 1 
                    for step in range(start_step, 19):
                        nr, nc = r + dr * step * sign, c + dc * step * sign
                        if not (0 <= nr < 19 and 0 <= nc < 19):
                            break
                        
                        stone = board[nr, nc]
                        if stone == player or stone == 3:
                            line_coords.append((nr, nc))
                            if stone == 3:
                                local_bonus += 1
                        else:
                            break
                
                if len(line_coords) >= 10:
                    clear_lines_count += 1
                    bonus_count += local_bonus
                    coords_to_clear.update(line_coords)

    if coords_to_clear:
        for r, c in coords_to_clear:
            new_board[r, c] = 0

    return (clear_lines_count * 20 + bonus_count * 3, new_board)

def check_connections(board: np.ndarray, r: int, c: int, player: int) -> Tuple[float, float, float]:
    '''Checks for connections (5+) and cleared lines (10+) from a given point.'''
    connected_lines = 0
    bonus_count = 0
    cleared_lines = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for dr, dc in directions:
        line_length = 1
        local_bonus = 0
        
        # Search in positive and negative directions
        for sign in [1, -1]:
            for step in range(1, 19):
                nr, nc = r + dr * step * sign, c + dc * step * sign
                if not (0 <= nr < 19 and 0 <= nc < 19):
                    break
                
                stone = board[nr, nc]
                if stone == player or stone == 3:
                    line_length += 1
                    if stone == 3:
                        local_bonus += 1
                else:
                    break

        if line_length >= 5:
            if line_length >= 10:
                cleared_lines += 1
            else:
                connected_lines += 1
            bonus_count += local_bonus

    return connected_lines, bonus_count, cleared_lines

# --- AI Interface Function ---

def get_board_tensor(game_state: GameState, network_type: str) -> np.ndarray:
    '''Generates the board tensor for the neural network.'''
    board = game_state["board"]
    current_player = game_state["current_player"]
    prev_actions = game_state["prev_actions"]
    stone_type = game_state["stone_type"]

    num_channels = 14 if network_type == 'policy' else 13
    board_tensor = np.zeros((num_channels, 19, 19), dtype=np.float32)

    board_tensor[0] = (board == 1)
    board_tensor[1] = (board == 2)
    board_tensor[2] = (board == 3)
    board_tensor[3] = 1.0  # Ones channel

    # Previous actions
    if prev_actions:
        player_turn = 3 - current_player
        for i, action in enumerate(prev_actions):
            for r, c in action:
                if 0 <= r < 19 and 0 <= c < 19:
                    board_tensor[4 + i, r, c] = player_turn
            player_turn = 3 - player_turn

    # Connection features
    for r, c in get_available_spaces(board):
        conn, bonus, clear = check_connections(board, r, c, current_player)
        board_tensor[7, r, c] = conn
        board_tensor[8, r, c] = bonus
        board_tensor[9, r, c] = clear

    # Zeros channel is already zero
    # board_tensor[10] = 0.0

    board_tensor[11] = 0.0 # Jama, seems unused in original logic
    board_tensor[12] = current_player

    if network_type == 'policy':
        board_tensor[13] = stone_type

    return board_tensor
