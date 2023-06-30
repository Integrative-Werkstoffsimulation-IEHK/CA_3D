# Create an empty multi-dimensional dictionary for the lookup table
import time
import numpy as np
# from utils.numba_functions import *
import numba
import gc
import math


user_input = {"start_real_radius": 0.5,
              "delta_radius": 0.1,
              "n_cells_per_axis": 501  # must be odd!!
              }


class Sphere:
    def __init__(self, params):
        self.ind_all = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],

                                 [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
                                 [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],

                                 [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
                                 [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_flat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)

        self.real_radius = params["start_real_radius"]
        self.delta_radius = params["delta_radius"]
        self.n_cells_per_axis = params["n_cells_per_axis"]

        self.s_coord = int((self.n_cells_per_axis - 1) / 2)
        # self.curr_side_lim = math.ceil(self.real_radius - 1 / 2)

        self.on_surface = np.empty((self.n_cells_per_axis ** 3, 3), dtype=np.short)
        self.on_surface[0] = [self.s_coord, self.s_coord, self.s_coord]
        self.last_on_surface_ind = 1

        self.shape = (self.n_cells_per_axis, self.n_cells_per_axis, self.n_cells_per_axis)
        self.c3d = np.array(np.full(self.shape, False, dtype=bool))
        self.c3d[self.s_coord, self.s_coord, self.s_coord] = True

        self.stats = {params["start_real_radius"]: {"mean": 0,
                                                    0: 0,
                                                    1: 0,
                                                    2: 0,
                                                    3: 0,
                                                    4: 0,
                                                    5: 0}}

        # self.inside = np.empty((self.n_cells_per_axis**3, 3), dtype=np.short)
        # self.last_inside_ind = 0

    def calc_all_sur_coord(self, seeds):
        around_seeds = np.array([item + self.ind_all for item in seeds], dtype=np.short)
        return around_seeds

    def calc_flat_sur_coord(self, seeds):
        around_seeds = np.array([item + self.ind_flat for item in seeds], dtype=np.short)
        return around_seeds

    def expand_radius(self):
        self.real_radius += self.delta_radius
        # self.curr_side_lim = math.ceil(self.real_radius - (1 / 2))
        # self.stats[self.real_radius] = {"mean": 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def calc_mean_neigh(self):
        self.expand_radius()
        arrounds = self.calc_all_sur_coord(self.on_surface[:self.last_on_surface_ind])

        self.last_on_surface_ind = check_coords_if_belong(self.c3d, arrounds, self.real_radius, self.s_coord, self.on_surface,
                                                  self.last_on_surface_ind)

        flat_arounds = self.calc_flat_sur_coord(self.on_surface[:self.last_on_surface_ind])
        neighbours = go_around_bool(self.c3d, flat_arounds)

        flat_arr_len = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)

        on_surface_ind = np.array(np.where(flat_arr_len < 6)[0])

        self.last_on_surface_ind = len(on_surface_ind)
        self.on_surface[:self.last_on_surface_ind] = self.on_surface[on_surface_ind]

        mean = np.mean(flat_arr_len[on_surface_ind])
        # unique_numbs = np.array(np.unique(flat_arr_len[on_surface_ind], return_counts=True))
        # freq = np.array(unique_numbs[1] / np.sum(unique_numbs[1]))
        print(self.real_radius, " ", mean)
        # return np.mean(on_surface), unique_numbs[0], freq
        # return np.mean(on_surface)


ind_formation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)


def calc_sur_coord(seeds):
    around_seeds = np.array([[item + ind_formation] for item in seeds], dtype=np.short)[:, 0]
    return around_seeds


@numba.njit(nopython=True)
def check_coords_if_belong(array_3d, arrounds, r, s_coord, on_surface, last_ind):
    for cell_arrounds in arrounds:
        for point in cell_arrounds:
            if (point[2] - s_coord) ** 2 + (point[1] - s_coord) ** 2 + (point[0] - s_coord) ** 2 <= r ** 2:
                if not array_3d[point[0], point[1], point[2]]:
                    array_3d[point[0], point[1], point[2]] = True
                    on_surface[last_ind] = [point[0], point[1], point[2]]
                    last_ind += 1
    return last_ind


@numba.njit(nopython=True)
def go_around_bool(array_3d, arrounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arrounds in arrounds:
        for point in seed_arrounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.bool_)


# @numba.njit(nopython=True)
def fill_sphere_coords(cells_3d, s_coord, r):
    all_where_true = [np.short([x, x, x]) for x in range(0)]
    for z in range(1, cells_3d.shape[0] - 1):
        for y in range(1, cells_3d.shape[0] - 1):
            for x in range(1, cells_3d.shape[0] - 1):
                if (x - s_coord) ** 2 + (y - s_coord) ** 2 + (z - s_coord) ** 2 <= r ** 2:
                    cells_3d[z, y, x] = True
                    all_where_true.append(np.short([z, y, x]))
    return all_where_true


def calc_mean_neigh(real_radius):
    # n_cells_per_axis = 2 * step + 3
    # real_radius = (2 * step + 1) / 2
    # shift = step + 1

    step = math.ceil(real_radius - 1 / 2)
    n_cells_per_axis = 2 * step + 3
    shift = step + 1

    shape = (n_cells_per_axis, n_cells_per_axis, n_cells_per_axis)
    c3d = np.array(np.full(shape, False, dtype=bool))

    belong_to_sphere = fill_sphere_coords(c3d, shift, real_radius)
    flat_arrounds = calc_sur_coord(belong_to_sphere)
    del belong_to_sphere
    gc.collect()

    neighbours = go_around_bool(c3d, flat_arrounds)
    del flat_arrounds, c3d
    gc.collect()

    flat_arr_len = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)

    inside_ind = np.array(np.where(flat_arr_len == 6)[0])
    on_surface = np.delete(flat_arr_len, inside_ind)

    # unique_numbs = np.array(np.unique(on_surface, return_counts=True))
    # freq = np.array(unique_numbs[1] / np.sum(unique_numbs[1]))

    # return np.mean(on_surface), unique_numbs[0], freq
    return np.mean(on_surface)


sphere = Sphere(user_input)

for _ in range(1000):
    sphere.calc_mean_neigh()

# init_rad = 0.5
# for s_step in range(100):
#     mean = calc_mean_neigh(init_rad)
#     init_rad += 0.1

#     print(f"""{s_step}   {mean}
# {unique}
# {fre}
# """)

