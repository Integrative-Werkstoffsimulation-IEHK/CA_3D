import multiprocessing
import gc
import numpy as np
from memory_profiler import profile


class CellularAutomata:
    @staticmethod
    @profile
    def worker(input_queue, output_queue):
        while True:
            try:
                args = input_queue.get()
                if args is None:  # Check for termination signal
                    break
                if args == "GC":
                    result = gc.collect()
                else:
                    callback = args[-1]
                    args = args[:-1]
                    result = callback(*args)

                output_queue.put(result)
            except Exception as e:
                print(f"Error in worker: {e}")
                output_queue.put(None)  # or handle the error appropriately

    def run_simulation(self):
        # Setup queues
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        # Start worker processes
        self.processes = []
        for _ in range(4):  # Adjust the number of worker processes as needed
            p = multiprocessing.Process(target=CellularAutomata.worker, args=(self.input_queue, self.output_queue))
            self.processes.append(p)
            p.start()

        # Dynamically feed new arguments to workers for this iteration
        for ind in self.comb_indexes:
            args = (self.product_x_nzs_mdata, self.primary_product_mdata, self.full_shm_mdata,
                    self.precip_3d_init_mdata, self.primary_active, self.primary_oxidant, [ind],
                    self.new_fetch_ind, self.nucleation_probabilities, CellularAutomata.ci_single_MP,
                    CellularAutomata.precip_step_standard_MP)
            self.input_queue.put(args)

        # Optionally, insert GC calls at appropriate intervals
        self.input_queue.put("GC")

        # Collect results for this iteration
        results = []
        for _ in self.comb_indexes:
            result = self.output_queue.get()
            if result is not None:
                results.append(result)

        # Signal workers to exit
        for _ in self.processes:
            self.input_queue.put(None)

        # Wait for all workers to finish
        for p in self.processes:
            p.join()

        return results

    # Example placeholders for class variables and methods
    comb_indexes = range(10)
    product_x_nzs_mdata = np.zeros((100, 100))  # Example data
    primary_product_mdata = np.zeros((100, 100))  # Example data
    full_shm_mdata = np.zeros((100, 100))  # Example data
    precip_3d_init_mdata = np.zeros((100, 100))  # Example data
    primary_active = np.zeros((100, 100))  # Example data
    primary_oxidant = np.zeros((100, 100))  # Example data
    new_fetch_ind = np.zeros((100, 100))  # Example data
    nucleation_probabilities = np.zeros((100, 100))  # Example data

    @staticmethod
    def ci_single_MP(*args):
        # Placeholder for the actual implementation
        return args

    @staticmethod
    def precip_step_standard_MP(*args):
        # Placeholder for the actual implementation
        return args

if __name__ == '__main__':
    ca = CellularAutomata()
    results = ca.run_simulation()
    print("Simulation results:", results)

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
