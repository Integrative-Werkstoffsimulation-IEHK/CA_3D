import numpy as np
import gc
import random
from microstructure import voronoi


class ActiveElem:
    def __init__(self, settings):
        self.cells_per_axis = settings["cells_per_axis"]
        self.neigh_range = settings["neigh_range"]
        self.shape = (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis)
        self.p1_range = settings["probabilities"][0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings["probabilities"][1]
        self.n_per_page = settings["n_per_page"]
        self.precip_transform_depth = int(10)  # min self.neigh_range !!!
        self.i_descards = None
        self.i_ind = None
        self.c3d = None
        self.cut_shape = None

        # exact concentration space fill _____________
        self.cells = np.array([[], [], []], dtype=np.short)
        for plane_xind in range(self.cells_per_axis):
            new_cells = np.array(random.sample(range(self.cells_per_axis**2), int(self.n_per_page)))
            new_cells = np.array(np.unravel_index(new_cells, (self.cells_per_axis, self.cells_per_axis)))
            new_cells = np.vstack((new_cells, np.full(len(new_cells[0]), plane_xind)))
            self.cells = np.concatenate((self.cells, new_cells), 1)
        self.cells = np.array(self.cells, dtype=np.short)
        # ____________________________________________

        # approx concentration space fill ____________
        # self.cells = np.random.randint(self.cells_per_axis, size=(3, int(self.n_per_page * self.cells_per_axis)),
        #                                dtype=np.short)
        # ____________________________________________

        # half space fill ____________________________
        # ind = np.where(self.cells[2] < int(self.cells_per_axis / 2))
        # self.cells = np.delete(self.cells, ind, 1)
        # ____________________________________________

        self.dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(self.cells[0]))
        self.dirs = np.array(np.unravel_index(self.dirs, (3, 3, 3)), dtype=np.byte)
        self.dirs -= 1

        self.current_count = self.n_per_page

    def diffuse(self):
        """
        Outgoing diffusion from the inside.
        """
        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(len(self.cells[0])), dtype=np.single)
        # randomise = np.array(np.random.random_sample(len(self.cells[0])))
        # deflection 1
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        # deflection 2
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        self.dirs[:, temp_ind] *= -1
        # deflection 3
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        # deflection 4
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        self.dirs[:, temp_ind] *= -1
        # reflection
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] *= -1
        del temp_ind
        gc.collect()

        self.cells = np.add(self.cells, self.dirs, casting="unsafe")

        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] == -1)[0]
        # closed left bound (reflection)
        self.cells[2, ind] = 1
        self.dirs[2, ind] = 1
        # _______________________
        # periodic____________________________________
        # self.cells[2, ind] = self.cells_per_axis - 1
        # ____________________________________________
        # open left bound___________________________
        # self.cells = np.delete(self.cells, ind, 1)
        # self.dirs = np.delete(self.dirs, ind, 1)
        # __________________________________________

        self.cells[0, np.where(self.cells[0] == -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] == self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] == -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] == self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] == self.cells_per_axis)[0]

        # closed right bound (reflection)____________
        self.cells[2, ind] = self.cells_per_axis - 2
        self.dirs[2, ind] = -1
        # ___________________________________________
        # open right bound___________________________
        # self.cells = np.delete(self.cells, ind, 1)
        # self.dirs = np.delete(self.dirs, ind, 1)
        # ___________________________________________
        # periodic____________________________________
        # self.cells[2, ind] = 0
        # ____________________________________________

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = self.n_cells_per_axis)
        self.current_count = len(np.where(self.cells[2] == self.cells_per_axis - 1)[0])
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_out_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page), dtype=np.short)
            new_out_page = np.concatenate((new_out_page, np.full((1, adj_cells_pro_page),
                                                                 self.cells_per_axis - 1, dtype=np.short)))
            # appending new generated particles as a ballistic ones to cells
            self.cells = np.concatenate((self.cells, new_out_page), axis=1)
            # appending new direction vectors to dirs
            new_dirs = np.zeros((3, adj_cells_pro_page), dtype=np.byte)
            new_dirs[2, :] = -1
            self.dirs = np.concatenate((self.dirs, new_dirs), axis=1)

    def transform_to_3d(self, furthest_i):
        if furthest_i + 1 + self.precip_transform_depth + self.neigh_range > self.cells_per_axis:
            depth = self.cells_per_axis + self.neigh_range
            last_i = self.cells_per_axis
        else:
            depth = furthest_i + 1 + self.precip_transform_depth + 1
            last_i = depth - 1

        self.cut_shape = (self.cells_per_axis, self.cells_per_axis, depth)
        self.c3d = np.full(self.cut_shape, 0, dtype=np.ubyte)
        self.i_ind = np.array(np.where(self.cells[2] < last_i)[0], dtype=np.uint32)
        self.i_descards = self.cells[:, self.i_ind]
        counts = np.unique(np.ravel_multi_index(self.i_descards, self.cut_shape), return_counts=True)
        dec = np.array(np.unravel_index(counts[0], self.cut_shape), dtype=np.short)
        counts = counts[1]
        self.c3d[dec[0], dec[1], dec[2]] = counts

    def transform_to_descards(self):
        values = np.array(self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]], dtype=np.ubyte)
        zero = np.array(np.where(values == 0)[0], dtype=np.uint32)
        ind_out = self.i_ind[zero]
        for step in range(np.max(values)):
            self.i_ind = np.delete(self.i_ind, zero)
            self.i_descards = np.delete(self.i_descards, zero, 1)

            u_dec = np.array(np.unique(np.ravel_multi_index(self.i_descards, self.cut_shape), return_index=True)[1])
            self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]] -= 1

            self.i_ind = np.delete(self.i_ind, u_dec)
            self.i_descards = np.delete(self.i_descards, u_dec, 1)

            values = self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]]
            zero = np.array(np.where(values == 0)[0], dtype=np.uint32)
            ind_out = np.concatenate((ind_out, self.i_ind[zero]))

        self.cells = np.delete(self.cells, ind_out, 1)
        self.dirs = np.delete(self.dirs, ind_out, 1)

        decomposed = np.array(np.nonzero(self.c3d), dtype=np.short)
        # print(len(decomposed[0]))
        if len(decomposed[0] > 0):
            # decomposed[2] -= 1
            self.cells = np.concatenate((self.cells, decomposed), axis=1)
            new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(decomposed[0]))
            new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
            new_dirs -= 1
            self.dirs = np.concatenate((self.dirs, new_dirs), axis=1)

        self.i_ind = None
        self.cut_shape = None
        self.i_descards = None
        self.c3d = None

    def count_cells_at_index(self, index):
        return len(np.where(self.cells[2] == index)[0])


class OxidantElem:
    def __init__(self, settings):
        self.cells_per_axis = settings["cells_per_axis"]
        self.p1_range = settings["probabilities"][0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings["probabilities"][1]
        self.n_per_page = settings["n_per_page"]
        self.current_count = 0
        self.furthest_index = None
        self.i_descards = None
        self.i_ind = None
        self.cut_shape = None
        self.c3d = None

        self.cells = np.array([[], [], []], dtype=np.short)
        self.dirs = np.zeros((3, len(self.cells[0])), dtype=np.byte)
        self.dirs[2] = 1
        self.current_count = 0
        self.fill_first_page()

        # self.microstructure = voronoi.VoronoiMicrostructure()
        # self.microstructure.generate_voronoi_3d(self.cells_per_axis, 5, seeds='standard')
        # self.microstructure.show_microstructure(self.cells_per_axis)
        # self.cross_shifts = np.array([[1, 0, 0], [0, 1, 0],
        #                               [-1, 0, 0], [0, -1, 0]], dtype=np.byte)

    def diffuse(self):
        """
        Outgoing diffusion from the inside.
        """

        # exists = self.microstructure.grain_boundaries[self.cells[0], self.cells[1], self.cells[2]]
        # # # print(exists)
        # temp_ind = np.array(np.where(exists)[0], dtype=np.uint32)
        # # print(temp_ind)
        # #
        # in_gb = np.array(self.cells[:, temp_ind], dtype=np.short)
        # # print(in_gb)
        # #
        # shift_vector = np.array(self.microstructure.jump_directions[in_gb[0], in_gb[1], in_gb[2]],
        #                         dtype=np.short).transpose()
        # # print(shift_vector)
        #
        # # print(self.cells)
        # cross_shifts = np.array(np.random.choice([0, 1, 2, 3], len(shift_vector[0])), dtype=np.ubyte)
        # cross_shifts = np.array(self.cross_shifts[cross_shifts], dtype=np.byte).transpose()
        #
        # shift_vector += cross_shifts
        #
        # self.cells[:, temp_ind] += shift_vector
        # # print(self.cells)

        # mixing particles according to Chopard and Droz
        randomise = np.array(np.random.random_sample(len(self.cells[0])), dtype=np.single)
        # deflection 1
        temp_ind = np.array(np.where(randomise <= self.p1_range)[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        temp_ind = np.array(np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 1, axis=0)
        self.dirs[:, temp_ind] *= -1
        temp_ind = np.array(np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        temp_ind = np.array(np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] = np.roll(self.dirs[:, temp_ind], 2, axis=0)
        self.dirs[:, temp_ind] *= -1
        temp_ind = np.array(np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0], dtype=np.uint32)
        self.dirs[:, temp_ind] *= -1
        del temp_ind
        gc.collect()
        self.cells = np.add(self.cells, self.dirs, casting="unsafe")
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.cells[2] <= -1)[0]
        self.cells[2, ind] = 0
        self.dirs[2, ind] = 1
        self.cells[0, np.where(self.cells[0] <= -1)] = self.cells_per_axis - 1
        self.cells[0, np.where(self.cells[0] >= self.cells_per_axis)] = 0
        self.cells[1, np.where(self.cells[1] <= -1)] = self.cells_per_axis - 1
        self.cells[1, np.where(self.cells[1] >= self.cells_per_axis)] = 0

        ind = np.where(self.cells[2] >= self.cells_per_axis)
        self.cells = np.delete(self.cells, ind, 1)
        self.dirs = np.delete(self.dirs, ind, 1)

        self.current_count = len(np.where(self.cells[2] == 0)[0])
        self.fill_first_page()

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = 0)
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_in_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page), dtype=np.short)
            new_in_page = np.concatenate((new_in_page, np.zeros((1, adj_cells_pro_page), dtype=np.short)))
            # appending new generated particles as a ballistic ones to cells1
            self.cells = np.concatenate((self.cells, new_in_page), axis=1)
            # appending new direction vectors to dirs
            new_dirs = np.zeros((3, adj_cells_pro_page), dtype=np.byte)
            new_dirs[2, :] = 1
            self.dirs = np.concatenate((self.dirs, new_dirs), axis=1)

    def transform_to_3d(self, furthest_i):
        self.cut_shape = (self.cells_per_axis, self.cells_per_axis, 1 + furthest_i + 1)
        self.c3d = np.full(self.cut_shape, 0, dtype=np.ubyte)
        self.i_ind = np.array(np.where(self.cells[2] <= furthest_i)[0], dtype=np.uint32)
        self.i_descards = self.cells[:, self.i_ind]
        counts = np.unique(np.ravel_multi_index(self.i_descards, self.cut_shape), return_counts=True)
        dec = np.array(np.unravel_index(counts[0], self.cut_shape), dtype=np.short)
        counts = counts[1]
        self.c3d[dec[0], dec[1], dec[2]] = counts

    def transform_to_descards(self):
        values = np.array(self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]], dtype=np.ubyte)
        zero = np.array(np.where(values == 0)[0], dtype=np.uint32)
        ind_out = self.i_ind[zero]
        for step in range(np.max(values, initial=0)):
            self.i_ind = np.delete(self.i_ind, zero)
            self.i_descards = np.delete(self.i_descards, zero, 1)
            u_dec = np.array(np.unique(np.ravel_multi_index(self.i_descards, self.cut_shape), return_index=True)[1])
            self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]] -= 1
            self.i_ind = np.delete(self.i_ind, u_dec)
            self.i_descards = np.delete(self.i_descards, u_dec, 1)
            values = self.c3d[self.i_descards[0], self.i_descards[1], self.i_descards[2]]
            zero = np.array(np.where(values == 0)[0], dtype=np.uint32)
            ind_out = np.concatenate((ind_out, self.i_ind[zero]))
        self.cells = np.delete(self.cells, ind_out, 1)
        self.dirs = np.delete(self.dirs, ind_out, 1)
        self.i_ind = None
        self.cut_shape = None
        self.i_descards = None
        self.c3d = None

    def count_cells_at_index(self, index):
        return len(np.where(self.cells[2] == index)[0])

    def calc_furthest_index(self):
        return np.amax(self.cells[2], initial=-1)


class Product:
    def __init__(self, settings):
        cells_per_axis = settings["cells_per_axis"]
        shape = (cells_per_axis, cells_per_axis, cells_per_axis)
        self.oxidation_number = settings["oxidation_number"]
        # self.oxidation_number = 1
        self.c3d = np.full(shape, 0, dtype=np.ubyte)
        self.full_c3d = np.full(shape, False)

    def transform_c3d(self):
        precipitations = np.array(np.nonzero(self.c3d), dtype=np.short)
        counts = self.c3d[precipitations[0], precipitations[1], precipitations[2]]
        return np.array(np.repeat(precipitations, counts, axis=1), dtype=np.short)

    def transform_single_c3d(self):
        return np.array(np.nonzero(self.c3d), dtype=np.short)

    def fix_full_cells(self, new_precip):
        current_precip = np.array(self.c3d[new_precip[0], new_precip[1], new_precip[2]])
        indexes = np.where(current_precip == self.oxidation_number)[0]
        full_precip = new_precip[:, indexes]
        self.full_c3d[full_precip[0], full_precip[1], full_precip[2]] = True
