# Create an empty multi-dimensional dictionary for the lookup table
import time
import numpy as np
import numba
from utils.numba_functions import *

ind_formation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)


def calc_sur_coord(seeds):
    around_seeds = np.array([[item + ind_formation] for item in seeds.transpose()], dtype=np.short)[:, 0]
    return around_seeds


@numba.njit(nopython=True)
def fill_sphere_coords(cells_3d, s_coord, r):
    max_coord = cells_3d.shape[0]

    # all_where_true = [np.short(x) for x in range(0)]

    for z in range(1, max_coord - 1):
        for y in range(1, max_coord - 1):
            for x in range(1, max_coord - 1):
                if (x - s_coord) ** 2 + (y - s_coord) ** 2 + (z - s_coord) ** 2 <= r ** 2:
                    cells_3d[z, y, x] = True
                    # all_where_true.append(np.short([z, y, x]))
    # return np.array(all_where_true, dtype=np.short)


def calc_mean_neigh(step):
    n_cells_per_axis = 2 * step + 3
    real_radius = (2 * step + 1) / 2
    shift = step + 1

    shape = (n_cells_per_axis, n_cells_per_axis, n_cells_per_axis)
    c3d = np.array(np.full(shape, False, dtype=bool))

    fill_sphere_coords(c3d, shift, real_radius)
    belong_to_sphere = np.array(np.nonzero(c3d))
    flat_arrounds = calc_sur_coord(belong_to_sphere)
    neighbours = go_around_bool(c3d, flat_arrounds)
    flat_arr_len = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
    inside_ind = np.array(np.where(flat_arr_len == 6)[0])
    on_surface = np.delete(flat_arr_len, inside_ind)

    return np.mean(on_surface)


for s_step in range(500):
    mean = calc_mean_neigh(s_step)
    print(f"""{s_step}  {mean}""")

