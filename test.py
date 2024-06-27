import multiprocessing
import time
import gc

some = gc.collect()

print()

# class DynamicProcessManager:
#     def __init__(self, num_workers):
#         self.num_workers = num_workers
#         self.input_queue = multiprocessing.Queue()
#         self.output_queue = multiprocessing.Queue()
#         self.workers = []
#         self._initialize_workers()
#
#     def _initialize_workers(self):
#         for _ in range(self.num_workers):
#             p = multiprocessing.Process(target=self.worker, args=(self.input_queue, self.output_queue))
#             p.start()
#             self.workers.append(p)
#
#     @staticmethod
#     def worker(input_queue, output_queue):
#         while True:
#             args = input_queue.get()
#             if args is None:  # Check for termination signal
#                 break
#             result = DynamicProcessManager.decomposition_intrinsic(*args)
#             output_queue.put(result)
#
#     @staticmethod
#     def decomposition_intrinsic(shape, shm):
#         # Your function implementation here
#         # This is just a placeholder for the actual work
#         time.sleep(1)  # Simulate work
#         return shape, shm
#
#     def process_iteration(self, chunk_ranges):
#         # Dynamically feed new arguments to workers for this iteration
#         for chunk_range in chunk_ranges:
#             args = (chunk_range['shape'], chunk_range['shm'])  # Replace with actual values for this iteration
#             self.input_queue.put(args)
#
#         # Collect results for this iteration
#         results = []
#         for _ in chunk_ranges:
#             result = self.output_queue.get()
#             results.append(result)
#
#         return results
#
#     def run_iterations(self, n_iterations, chunk_ranges_list):
#         for iteration in range(n_iterations):
#             print(f"Iteration {iteration + 1}")
#             chunk_ranges = chunk_ranges_list[iteration]
#
#             results = self.process_iteration(chunk_ranges)
#             print(f"Results for iteration {iteration + 1}: {results}")
#
#         self.terminate_workers()
#
#     def terminate_workers(self):
#         # Signal workers to terminate
#         for _ in self.workers:
#             self.input_queue.put(None)
#
#         # Wait for all workers to terminate
#         for worker in self.workers:
#             worker.join()
#
#         print("All iterations completed.")
#
# if __name__ == '__main__':
#     num_workers = 4  # Adjust based on your requirements
#     manager = DynamicProcessManager(num_workers)
#
#     n_iterations = 10  # Number of iterations
#     chunk_ranges_list = [
#         [{'shape': (i, i+1), 'shm': f'shm_{i}'} for i in range(5)] for _ in range(n_iterations)
#     ]  # Replace with actual chunk_ranges for each iteration
#
#     manager.run_iterations(n_iterations, chunk_ranges_list)
