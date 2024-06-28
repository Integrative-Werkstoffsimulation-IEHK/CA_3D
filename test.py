import multiprocessing
from multiprocessing import shared_memory
import numpy as np

import utils
from configuration import Config
import numba

SHAPE = (10000, 10000)
DTYPE = int

N_PROC = 10
N_TASKS = 10
N_ITER = 10000


@numba.njit
def increase_counts(array_2d):
    for ind_x in range(array_2d.shape[0]):
        for ind_y in range(array_2d.shape[1]):
            array_2d[ind_x, ind_y] += 1


class Other:
    def __init__(self):
        some_huge_shit = np.zeros(SHAPE, dtype=DTYPE)
        self.huge_shit_shm = shared_memory.SharedMemory(create=True, size=some_huge_shit.nbytes)

        self.huge_shit = np.ndarray(some_huge_shit.shape, dtype=some_huge_shit.dtype, buffer=self.huge_shit_shm.buf)
        np.copyto(self.huge_shit, some_huge_shit)

        self.huge_shit_mdata = SharedMetaData(self.huge_shit_shm.name, self.huge_shit.shape, self.huge_shit.dtype)

    def do_in_parent(self):
        increase_counts(self.huge_shit)
        print("Done In Parent")


class SharedMetaData:
    def __init__(self, shm_name, shape, dtype):
        self.name = shm_name
        self.shape = shape
        self.dtype = dtype


def worker(args):
    callback = args[-1]
    args = args[:-1]
    result = callback(*args)
    return result


def heavy_work(shm_mdata, huge_shit_mdata, probs, conf):
    probab = probs
    cfeg = conf

    shm_o = shared_memory.SharedMemory(name=shm_mdata.name)
    my_array = np.ndarray(shm_mdata.shape, dtype=shm_mdata.dtype, buffer=shm_o.buf)

    shm_other = shared_memory.SharedMemory(name=huge_shit_mdata.name)
    other_array = np.ndarray(huge_shit_mdata.shape, dtype=huge_shit_mdata.dtype, buffer=shm_other.buf)

    increase_counts(my_array)
    increase_counts(other_array)

    shm_o.close()
    shm_other.close()
    return 0


class MyPool:
    def __init__(self):
        self.pool = multiprocessing.Pool(processes=N_PROC)

        some_huge_shit = np.zeros(SHAPE, dtype=DTYPE)
        self.huge_shit_shm = shared_memory.SharedMemory(create=True, size=some_huge_shit.nbytes)

        huge_shit = np.ndarray(some_huge_shit.shape, dtype=some_huge_shit.dtype, buffer=self.huge_shit_shm.buf)
        np.copyto(huge_shit, some_huge_shit)

        self.huge_shit_mdata = SharedMetaData(self.huge_shit_shm.name, huge_shit.shape, huge_shit.dtype)

        self.config = Config()

        self.other = Other()

    def start_pool(self):
        probs = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                 Config.PRODUCTS.PRIMARY)
        tasks = [(self.huge_shit_mdata, self.other.huge_shit_mdata, self.config, probs, heavy_work) for _ in range(N_TASKS)]
        self.pool.map(worker, tasks)

    def terminate_workers(self):
        self.pool.close()
        self.pool.join()

        self.huge_shit_shm.close()
        self.huge_shit_shm.unlink()


if __name__ == '__main__':
    new_utils = utils.Utils()
    new_utils.generate_param()



    new_pool = MyPool()

    for iteration in range(N_ITER):
        print(iteration)
        new_pool.start_pool()
        new_pool.other.do_in_parent()
