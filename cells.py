import time

import numpy as np
import gc


class ActiveElem:
    def __init__(self, settings):
        self.cells_per_axis = settings["cells_per_axis"]
        self.p1_range = settings["probabilities"][0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings["probabilities"][1]

        self.n_per_page = settings["n_per_page"]
        self.c1 = np.random.randint(self.cells_per_axis, size=(3, int(self.n_per_page * self.cells_per_axis)),
                                    dtype=np.short)
        self.c2 = np.array([[], [], []], dtype=np.short)
        self.c3 = np.array([[], [], []], dtype=np.short)
        self.c4 = np.array([[], [], []], dtype=np.short)
        self.c5 = np.array([[], [], []], dtype=np.short)
        self.c6 = np.array([[], [], []], dtype=np.short)
        self.t1 = np.array([[], [], []], dtype=np.short)
        self.t2 = np.array([[], [], []], dtype=np.short)
        self.t3 = np.array([[], [], []], dtype=np.short)
        self.t4 = np.array([[], [], []], dtype=np.short)
        self.t5 = np.array([[], [], []], dtype=np.short)
        self.t6 = np.array([[], [], []], dtype=np.short)
        self.current_count = self.n_per_page

    def diffuse(self):
        """
        Outgoing diffusion from the inside.
        """
        # mixing particles according to Chopard and Droz
        #  -------------------------------------------------------------------------------------------------------------
        # cell 1
        randomise = np.random.random_sample(len(self.c1[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c1[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c1[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c1[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c1[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t3 = np.concatenate((self.t3, self.c1[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t1 = np.concatenate((self.t1, self.c1[:, temp_ind]), axis=1)
        # self.c1 = np.array([[], [], []], dtype=int)
        # cell 2
        randomise = np.random.random_sample(len(self.c2[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t1 = np.concatenate((self.t1, self.c2[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c2[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t3 = np.concatenate((self.t3, self.c2[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c2[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t5 = np.concatenate((self.t5, self.c2[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t2 = np.concatenate((self.t2, self.c2[:, temp_ind]), axis=1)
        # self.c2 = np.array([[], [], []], dtype=int)
        # cell 3
        randomise = np.random.random_sample(len(self.c3[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c3[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c3[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c3[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c3[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t1 = np.concatenate((self.t1, self.c3[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t3 = np.concatenate((self.t3, self.c3[:, temp_ind]), axis=1)
        # self.c3 = np.array([[], [], []], dtype=int)
        # cell 4
        randomise = np.random.random_sample(len(self.c4[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c4[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t1 = np.concatenate((self.t1, self.c4[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c4[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t3 = np.concatenate((self.t3, self.c4[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t6 = np.concatenate((self.t6, self.c4[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t4 = np.concatenate((self.t4, self.c4[:, temp_ind]), axis=1)
        # self.c4 = np.array([[], [], []], dtype=int)
        # cell 5
        randomise = np.random.random_sample(len(self.c5[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t1 = np.concatenate((self.t1, self.c5[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c5[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t3 = np.concatenate((self.t3, self.c5[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c5[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t2 = np.concatenate((self.t2, self.c5[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t5 = np.concatenate((self.t5, self.c5[:, temp_ind]), axis=1)
        # self.c5 = np.array([[], [], []], dtype=int)
        # cell 6
        randomise = np.random.random_sample(len(self.c6[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c6[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t1 = np.concatenate((self.t1, self.c6[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c6[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t3 = np.concatenate((self.t3, self.c6[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t4 = np.concatenate((self.t4, self.c6[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t6 = np.concatenate((self.t6, self.c6[:, temp_ind]), axis=1)
        # self.c6 = np.array([[], [], []], dtype=int)
        del temp_ind
        gc.collect()
        #  -------------------------------------------------------------------------------------------------------------
        # adjusting a coordinates of side points for correct shifting
        ind = np.where(self.t1[2, :] == 0)[0]
        reflected = self.t1[:, ind]
        self.t1 = np.delete(self.t1, ind, 1)
        self.t3 = np.delete(self.t3, np.where(self.t3[2, :] == self.cells_per_axis - 1)[0], 1)
        self.t3 = np.concatenate((self.t3, reflected), axis=1)
        self.t2[1, np.where(self.t2[1, :] == self.cells_per_axis - 1)] = -1
        self.t4[0, np.where(self.t4[0, :] == self.cells_per_axis - 1)] = -1
        self.t5[1, np.where(self.t5[1, :] == 0)] = self.cells_per_axis
        self.t6[0, np.where(self.t6[0, :] == 0)] = self.cells_per_axis
        # applying a shift, direction and periodic boundary conditions respectively
        self.t1[2, :] -= 1
        self.t2[1, :] += 1
        self.t3[2, :] += 1
        self.t4[0, :] += 1
        self.t5[1, :] -= 1
        self.t6[0, :] -= 1
        #  -------------------------------------------------------------------------------------------------------------
        # counting a number of particles at the 0's page for keeping its number constant!
        #  -------------------------------------------------------------------------------------------------------------
        self.current_count = len(np.where(self.t2[2, :] == self.cells_per_axis - 1)[0]) \
                             + len(np.where(self.t3[2, :] == self.cells_per_axis - 1)[0]) \
                             + len(np.where(self.t4[2, :] == self.cells_per_axis - 1)[0]) \
                             + len(np.where(self.t5[2, :] == self.cells_per_axis - 1)[0]) \
                             + len(np.where(self.t6[2, :] == self.cells_per_axis - 1)[0])
        #  -------------------------------------------------------------------------------------------------------------
        # relocating particles back to initial cells and resetting temporary containers
        #  -------------------------------------------------------------------------------------------------------------
        self.c1 = self.t1
        self.t1 = np.array([[], [], []], dtype=np.short)
        self.c2 = self.t2
        self.t2 = np.array([[], [], []], dtype=np.short)
        self.c3 = self.t3
        self.t3 = np.array([[], [], []], dtype=np.short)
        self.c4 = self.t4
        self.t4 = np.array([[], [], []], dtype=np.short)
        self.c5 = self.t5
        self.t5 = np.array([[], [], []], dtype=np.short)
        self.c6 = self.t6
        self.t6 = np.array([[], [], []], dtype=np.short)
        #  -------------------------------------------------------------------------------------------------------------
        self.fill_first_page()

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = self.n_cells_per_axis)
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_out_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page))
            new_out_page = np.concatenate((new_out_page, np.full((1, adj_cells_pro_page),
                                                                 self.cells_per_axis - 1, dtype=np.short)))
            # appending new generated particles as a ballistic ones to cells1
            self.c1 = np.concatenate((self.c1, new_out_page), axis=1)

    def sum_up_cells(self):
        return np.concatenate((self.c1, self.c2, self.c3, self.c4, self.c5, self.c6), axis=1)

    def count_cells_at_index(self, index):
        cells = self.sum_up_cells()
        return len(np.where(cells[2] == index)[0])


class OxidantElem:
    def __init__(self, settings):
        self.cells_per_axis = settings["cells_per_axis"]
        self.p1_range = settings["probabilities"][0]
        self.p2_range = 2 * self.p1_range
        self.p3_range = 3 * self.p1_range
        self.p4_range = 4 * self.p1_range
        self.p_r_range = self.p4_range + settings["probabilities"][1]
        self.n_per_page = settings["n_per_page"]
        self.c1 = np.array([[], [], []], dtype=int)
        self.c2 = np.array([[], [], []], dtype=int)
        self.c3 = np.array([[], [], []], dtype=int)
        self.c4 = np.array([[], [], []], dtype=int)
        self.c5 = np.array([[], [], []], dtype=int)
        self.c6 = np.array([[], [], []], dtype=int)
        self.t1 = np.array([[], [], []], dtype=int)
        self.t2 = np.array([[], [], []], dtype=int)
        self.t3 = np.array([[], [], []], dtype=int)
        self.t4 = np.array([[], [], []], dtype=int)
        self.t5 = np.array([[], [], []], dtype=int)
        self.t6 = np.array([[], [], []], dtype=int)
        self.current_count = 0
        self.furthest_index = None
        self.fill_first_page()

    def diffuse(self):
        """
        Outgoing diffusion from the inside.
        """
        # mixing particles according to Chopard and Droz
        #  -------------------------------------------------------------------------------------------------------------
        # cell 1
        randomise = np.random.random_sample(len(self.c1[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c1[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c1[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c1[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c1[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t3 = np.concatenate((self.t3, self.c1[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t1 = np.concatenate((self.t1, self.c1[:, temp_ind]), axis=1)
        # self.c1 = np.array([[], [], []], dtype=int)
        # cell 2
        randomise = np.random.random_sample(len(self.c2[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t1 = np.concatenate((self.t1, self.c2[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c2[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t3 = np.concatenate((self.t3, self.c2[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c2[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t5 = np.concatenate((self.t5, self.c2[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t2 = np.concatenate((self.t2, self.c2[:, temp_ind]), axis=1)
        # self.c2 = np.array([[], [], []], dtype=int)
        # cell 3
        randomise = np.random.random_sample(len(self.c3[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c3[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c3[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c3[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c3[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t1 = np.concatenate((self.t1, self.c3[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t3 = np.concatenate((self.t3, self.c3[:, temp_ind]), axis=1)
        # self.c3 = np.array([[], [], []], dtype=int)
        # cell 4
        randomise = np.random.random_sample(len(self.c4[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c4[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t1 = np.concatenate((self.t1, self.c4[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c4[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t3 = np.concatenate((self.t3, self.c4[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t6 = np.concatenate((self.t6, self.c4[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t4 = np.concatenate((self.t4, self.c4[:, temp_ind]), axis=1)
        # self.c4 = np.array([[], [], []], dtype=int)
        # cell 5
        randomise = np.random.random_sample(len(self.c5[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t1 = np.concatenate((self.t1, self.c5[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t4 = np.concatenate((self.t4, self.c5[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t3 = np.concatenate((self.t3, self.c5[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t6 = np.concatenate((self.t6, self.c5[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t2 = np.concatenate((self.t2, self.c5[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t5 = np.concatenate((self.t5, self.c5[:, temp_ind]), axis=1)
        # self.c5 = np.array([[], [], []], dtype=int)
        # cell 6
        randomise = np.random.random_sample(len(self.c6[0]))
        # deflection 1
        temp_ind = np.where(randomise <= self.p1_range)[0]
        self.t2 = np.concatenate((self.t2, self.c6[:, temp_ind]), axis=1)
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))[0]
        self.t1 = np.concatenate((self.t1, self.c6[:, temp_ind]), axis=1)
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))[0]
        self.t5 = np.concatenate((self.t5, self.c6[:, temp_ind]), axis=1)
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))[0]
        self.t3 = np.concatenate((self.t3, self.c6[:, temp_ind]), axis=1)
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))[0]
        self.t4 = np.concatenate((self.t4, self.c6[:, temp_ind]), axis=1)
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)[0]
        self.t6 = np.concatenate((self.t6, self.c6[:, temp_ind]), axis=1)
        # self.c6 = np.array([[], [], []], dtype=int)
        del temp_ind
        gc.collect()
        #  -------------------------------------------------------------------------------------------------------------
        # adjusting a coordinates of side points for correct shifting
        self.t1 = np.delete(self.t1, np.where(self.t1[2, :] == self.cells_per_axis - 1), 1)
        self.t3 = np.delete(self.t3, np.where(self.t3[2, :] == 0)[0], 1)
        self.t2[1, np.where(self.t2[1, :] == self.cells_per_axis - 1)] = -1
        self.t4[0, np.where(self.t4[0, :] == self.cells_per_axis - 1)] = -1
        self.t5[1, np.where(self.t5[1, :] == 0)] = self.cells_per_axis
        self.t6[0, np.where(self.t6[0, :] == 0)] = self.cells_per_axis
        # applying a shift, direction and periodic boundary conditions respectively
        self.t1[2, :] += 1
        self.t2[1, :] += 1
        self.t3[2, :] -= 1
        self.t4[0, :] += 1
        self.t5[1, :] -= 1
        self.t6[0, :] -= 1
        #  -------------------------------------------------------------------------------------------------------------
        # counting a number of particles at the 0's page for keeping its number constant!
        #  -------------------------------------------------------------------------------------------------------------
        self.current_count = len(np.where(self.t2[2, :] == 0)[0]) \
                             + len(np.where(self.t3[2, :] == 0)[0]) \
                             + len(np.where(self.t4[2, :] == 0)[0]) \
                             + len(np.where(self.t5[2, :] == 0)[0]) \
                             + len(np.where(self.t6[2, :] == 0)[0])
        #  -------------------------------------------------------------------------------------------------------------
        # relocating particles back to initial cells and resetting temporary containers
        #  -------------------------------------------------------------------------------------------------------------
        self.c1 = self.t1
        self.t1 = np.array([[], [], []], dtype=int)
        self.c2 = self.t2
        self.t2 = np.array([[], [], []], dtype=int)
        self.c3 = self.t3
        self.t3 = np.array([[], [], []], dtype=int)
        self.c4 = self.t4
        self.t4 = np.array([[], [], []], dtype=int)
        self.c5 = self.t5
        self.t5 = np.array([[], [], []], dtype=int)
        self.c6 = self.t6
        self.t6 = np.array([[], [], []], dtype=int)
        #  -------------------------------------------------------------------------------------------------------------
        self.fill_first_page()

    def fill_first_page(self):
        # generating new particles on the diffusion surface (X = 0)
        adj_cells_pro_page = self.n_per_page - self.current_count
        if adj_cells_pro_page > 0:
            new_in_page = np.random.randint(self.cells_per_axis, size=(2, adj_cells_pro_page))
            new_in_page = np.concatenate((new_in_page, np.zeros((1, adj_cells_pro_page), dtype=int)))
            # appending new generated particles as a ballistic ones to cells1
            self.c1 = np.concatenate((self.c1, new_in_page), axis=1)

    def sum_up_cells(self):
        return np.concatenate((self.c1, self.c2, self.c3, self.c4, self.c5, self.c6), axis=1)

    def count_cells_at_index(self, index):
        cells = self.sum_up_cells()
        return len(np.where(cells[2] == index)[0])

    def calc_furthest_index(self):
        return np.amax([np.amax(self.c1[2], initial=-1), np.amax(self.c2[2], initial=-1),
                        np.amax(self.c3[2], initial=-1), np.amax(self.c4[2], initial=-1),
                        np.amax(self.c5[2], initial=-1), np.amax(self.c6[2], initial=-1)])


class Product:
    def __init__(self, shape):
        self.c3d = np.full(shape, False)

    def transform_c3d(self):
        return np.array(np.nonzero(self.c3d), dtype=int)


class ProductInt:
    def __init__(self, settings):
        cells_per_axis = settings["cells_per_axis"]
        shape = (cells_per_axis, cells_per_axis, cells_per_axis)
        self.oxidation_number = settings["oxidation_number"]
        self.c3d = np.full(shape, 0, dtype=int)
        self.full_c3d = np.full(shape, False)

    def transform_c3d(self):
        precipitations = np.array(np.nonzero(self.c3d), dtype=int)
        counts = self.c3d[precipitations[0], precipitations[1], precipitations[2]]
        return np.repeat(precipitations, counts, axis=1)

    def fix_full_cells(self, new_precip, plane_index):
        current_precip = np.array(self.c3d[new_precip[0], new_precip[1], plane_index])
        indexes = np.where(current_precip == self.oxidation_number)[0]
        full_precip = new_precip[:, indexes]
        self.full_c3d[full_precip[0], full_precip[1], plane_index] = True


