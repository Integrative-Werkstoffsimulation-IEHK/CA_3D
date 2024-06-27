import numpy as np
from configuration import Config


neigh_range = Config.NEIGH_RANGE
n_cells_per_axis = Config.N_CELLS_PER_AXIS


def generate_neigh_indexes_flat():
    size = 3 + (neigh_range - 1) * 2
    neigh_shape = (size, size, 3)
    temp = np.ones(neigh_shape, dtype=int)
    temp[:, :, 0] = 0
    temp[:, :, 2] = 0

    flat_ind = np.array(ind_decompose_flat_z)
    flat_ind = flat_ind.transpose()
    flat_ind[0] += neigh_range
    flat_ind[1] += neigh_range
    flat_ind[2] += 1

    temp[flat_ind[0], flat_ind[1], flat_ind[2]] = 0

    coord = np.array(np.nonzero(temp))
    coord[0] -= neigh_range
    coord[1] -= neigh_range
    coord[2] -= 1
    coord = coord.transpose()

    coord = np.concatenate((ind_decompose_flat_z, coord))
    additional = coord[[2, 5]]
    coord = np.delete(coord, [2, 5], axis=0)
    coord = np.concatenate((coord, additional))

    return np.array(coord, dtype=np.byte)


ind_decompose_flat_z = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)

ind_decompose_no_flat = np.array(
            [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

ind_formation = generate_neigh_indexes_flat()


def calc_sur_ind_decompose_flat_with_zero(seeds):
    seeds = seeds.transpose()
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_decompose_flat_z] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    around_seeds[around_seeds == n_cells_per_axis] = 0
    around_seeds[around_seeds == -1] = n_cells_per_axis - 1
    return around_seeds

def calc_sur_ind_decompose_no_flat(seeds):
    seeds = seeds.transpose()
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_decompose_no_flat] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    around_seeds[around_seeds == n_cells_per_axis] = 0
    around_seeds[around_seeds == -1] = n_cells_per_axis - 1
    return around_seeds

def calc_sur_ind_formation(seeds, dummy_ind):
    # generating a neighbouring coordinates for each seed (including the position of the seed itself)
    around_seeds = np.array([[item + ind_formation] for item in seeds], dtype=np.short)[:, 0]
    # applying periodic boundary conditions
    if seeds[0, 2] < neigh_range:
        indexes = np.where(around_seeds[:, :, 2] < 0)
        around_seeds[indexes[0], indexes[1], 2] = dummy_ind
    for shift in range(neigh_range):
        indexes = np.where(around_seeds[:, :, 0:2] == n_cells_per_axis + shift)
        around_seeds[indexes[0], indexes[1], indexes[2]] = shift
        indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
        around_seeds[indexes[0], indexes[1], indexes[2]] = n_cells_per_axis - shift - 1
    return around_seeds


