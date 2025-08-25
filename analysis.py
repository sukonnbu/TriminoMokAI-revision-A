'''
This script benchmarks the performance of the object-oriented vs. functional
implementations of the TriminoMok game and MCTS algorithm.

It compares:
1. Average time taken for an MCTS search.
2. Memory usage during a short game simulation.

To run this script and get memory usage, you need to have memory_profiler installed:
`pip install memory-profiler`

Then, run it from your terminal like this:
`python -m memory_profiler analysis.py`
'''

import time
import torch
import numpy as np
from memory_profiler import profile

# --- Import Original Object-Oriented Components ---
from cnn import PolicyNetwork, ValueNetwork
from mcts import TriminoMok as OopTriminoMok, Mcts as OopMcts
from main import init_board as oop_init_board

# --- Import New Functional Components ---
import functional_triminomok as ft
from functional_mcts import Mcts as FunctionalMcts
from functional_main import init_board as functional_init_board

# --- Benchmark Configuration ---
MODEL_NUM = 130
NUM_TURNS_TO_SIMULATE = 15  # Run for a fixed number of turns for a fair comparison
NUM_MCTS_ITERATIONS = 30 # Use fewer iterations for a quicker analysis

def load_models():
    '''Loads the same pre-trained models for both tests.'''
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    try:
        policy_net.load_state_dict(torch.load(f"./saves/setting_1/policy_net_{MODEL_NUM}.pth"))
        value_net.load_state_dict(torch.load(f"./saves/setting_1/value_net_{MODEL_NUM}.pth"))
    except FileNotFoundError:
        print("Model files not found. Please ensure models are available.")
        return None, None
    policy_net.eval()
    value_net.eval()
    return policy_net, value_net

# The @profile decorator will measure the memory consumption of this function.
@profile
def run_oop_simulation(policy_net, value_net):
    '''Runs the game simulation using the Object-Oriented approach.'''
    print("--- Running Object-Oriented Simulation ---")
    board, initial_stones, stone_type = oop_init_board()
    game_state = OopTriminoMok(board, stone_type, initial_stones, depth=1)
    mcts = OopMcts(policy_net, value_net)
    
    timings = []
    for _ in range(NUM_TURNS_TO_SIMULATE):
        if game_state.is_terminal(40):
            break
        
        start_time = time.perf_counter()
        best_move = mcts.run(game_state, NUM_MCTS_ITERATIONS, 40)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        
        if best_move == (0, 0, 0):
            break
        game_state.make_move(best_move)
        
    avg_time = np.mean(timings) if timings else 0
    print(f"Average MCTS search time: {avg_time:.4f} seconds")
    return avg_time

# The @profile decorator will measure the memory consumption of this function.
@profile
def run_functional_simulation(policy_net, value_net):
    '''Runs the game simulation using the Functional Programming approach.'''
    print("\n--- Running Functional Simulation ---")
    game_state = functional_init_board()
    mcts = FunctionalMcts(policy_net, value_net)
    
    timings = []
    for _ in range(NUM_TURNS_TO_SIMULATE):
        if ft.is_terminal(game_state, 40):
            break
            
        start_time = time.perf_counter()
        best_move = mcts.run(game_state, NUM_MCTS_ITERATIONS, 40)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)

        if best_move == (0, 0, 0):
            break
        # Re-assign the state variable with the new state returned by the pure function
        game_state = ft.make_move(game_state, best_move)

    avg_time = np.mean(timings) if timings else 0
    print(f"Average MCTS search time: {avg_time:.4f} seconds")
    return avg_time


def main():
    '''Main function to run and print the benchmark analysis.'''
    policy_net, value_net = load_models()
    if policy_net is None:
        return

    # Run simulations
    oop_avg_time = run_oop_simulation(policy_net, value_net)
    functional_avg_time = run_functional_simulation(policy_net, value_net)

    # Print summary
    print("\n--- Benchmark Summary ---")
    print(f"Object-Oriented Avg. Time:   {oop_avg_time:.4f}s")
    print(f"Functional Avg. Time:        {functional_avg_time:.4f}s")
    
    time_diff = functional_avg_time - oop_avg_time
    time_diff_percent = (time_diff / oop_avg_time) * 100 if oop_avg_time != 0 else 0

    print("\nConclusion:")
    print("The memory usage for each approach is detailed above the respective sections.")
    if abs(time_diff_percent) < 5:
        print("Execution times are very similar (less than 5% difference).")
    elif time_diff > 0:
        print(f"The functional approach was approximately {time_diff_percent:.2f}% slower.")
        print("This could be due to the overhead of creating new state objects (e.g., deep copying the board) in each step instead of modifying them in place.")
    else:
        print(f"The functional approach was approximately {-time_diff_percent:.2f}% faster.")
        print("This could be due to various factors like more efficient data access or Python's optimization of certain function calls.")

if __name__ == "__main__":
    main()
