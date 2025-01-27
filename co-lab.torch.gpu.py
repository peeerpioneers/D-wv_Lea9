import torch
import time
import numpy as np
import torch.multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_operation(size):
    """Performs a CPU-intensive calculation using a tensor."""
    a = torch.rand(size, size) # create a random CPU tensor

    start_time = time.time()
    # Perform a sequence of CPU-heavy operations.
    for _ in range(50):
      a = torch.cos(a) # cosine
      a = torch.sqrt(torch.abs(a)) # absolute value and square root
      a = torch.log10(torch.abs(a)) # absolute value and log10
      a = torch.sin(a) # sine

    end_time = time.time()
    return end_time - start_time

def worker_function(size, repetitions):
    """Worker function that calls a cpu bound operation repeatedly"""
    total_time = 0
    for _ in range(repetitions):
      total_time += cpu_bound_operation(size)

    return total_time

def stress_cpu_multiprocessing(size, num_processes, repetitions):
  """Uses multiprocessing to stress the CPU cores."""
  print(f"Testing multiprocessing CPU with tensors of size ({size}, {size})")
  start_time = time.time()

  with ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = executor.map(worker_function, [size]*num_processes, [repetitions]*num_processes)
    total_time = sum(results)

  end_time = time.time()
  print(f"Multiprocessing CPU operation took {end_time-start_time:.4f} seconds. Total computation time in all processes: {total_time}")
  return end_time-start_time

def run_tests(size_mult=1):
    """Runs all tests."""
    results = []
    size = int(128*size_mult)
    repetitions=int(1000/size_mult)
    num_processes = os.cpu_count() # check number of cores available

    cpu_time = stress_cpu_multiprocessing(size, num_processes, repetitions)
    results.append({"test": "multiprocessing CPU", "size": size, "num_processes":num_processes, "repetitions":repetitions, "time": cpu_time})

    return results


if __name__ == '__main__':
    # sizes is multiplied by size_mult to have larger stress tests.
    # Start with small numbers, then increase
    size_multipliers = [1, 2, 4, 8, 16]
    all_results = []
    for size_mult in size_multipliers:
      print("------------------------------------------------------------")
      print(f"Running tests with size multiplier: {size_mult}")
      results = run_tests(size_mult=size_mult)
      all_results.append({"size_multiplier": size_mult, "results": results})

    import json
    with open('cpu_stress_test_results.json', 'w') as f:
      json.dump(all_results, f, indent=4)

    print("CPU stress test complete. Results saved to cpu_stress_test_results.json")
