from utils.utilities import *
import gc
import progressbar
# import numpy as np
# import time
import multiprocessing


class CellularAutomata:
    def __init__(self, param=None):
        # pre settings
        if param is None:
            # setting default parameters if no param is given
            param = physical_data.DEFAULT_PARAM
        self.param = param
        self.utils = Utils(self.param)
        self.save_whole = self.param["save_whole"]
        self.begin = time.time()
        self.elapsed_time = 0

        # simulated space parameters
        self.n_cells_per_axis = self.param["n_cells_per_axis"]
        self.n_iterations = self.param["n_iterations"]
        self.shape = (self.n_cells_per_axis, self.n_cells_per_axis, self.n_cells_per_axis)

        # setting variables for inward diffusion
        if self.param["inward_diffusion"]:
            self.p_in = self.utils.p_in
            self.p1_range = 1 * self.p_in["p"]
            self.p2_range = 2 * self.p_in["p"]
            self.p3_range = 3 * self.p_in["p"]
            self.p4_range = 4 * self.p_in["p"]
            self.p_r_range = self.p4_range + self.p_in["p3"]
            self.number_inward_cells_pro_page = int(self.param["diff_elem_conc"] * self.n_cells_per_axis**2)
            self.cells1 = np.array([], dtype=int)
            self.cells2 = np.array([], dtype=int)
            self.cells3 = np.array([], dtype=int)
            self.cells4 = np.array([], dtype=int)
            self.cells5 = np.array([], dtype=int)
            self.cells6 = np.array([], dtype=int)
            self.cells1_contain = np.array([], dtype=int)
            self.cells2_contain = np.array([], dtype=int)
            self.cells3_contain = np.array([], dtype=int)
            self.cells4_contain = np.array([], dtype=int)
            self.cells5_contain = np.array([], dtype=int)
            self.cells6_contain = np.array([], dtype=int)
            self.inward = None
            self.furthest_inward_x = None

        # setting variables for outward diffusion
        if self.param["outward_diffusion"]:
            self.p_out = self.utils.p_out
            self.p1_range_out = 1 * self.p_out["p"]
            self.p2_range_out = 2 * self.p_out["p"]
            self.p3_range_out = 3 * self.p_out["p"]
            self.p4_range_out = 4 * self.p_out["p"]
            self.p_r_range_out = self.p4_range_out + self.p_out["p3"]
            self.number_outward_cells_pro_page = int(self.param["active_elem_conc"] * 0.18 * self.n_cells_per_axis ** 2)
            self.cells1_out = np.random.randint(self.n_cells_per_axis**3 - 1,
                                                size=int(self.param["active_elem_conc"] * self.n_cells_per_axis ** 3))
            self.cells2_out = np.array([], dtype=int)
            self.cells3_out = np.array([], dtype=int)
            self.cells4_out = np.array([], dtype=int)
            self.cells5_out = np.array([], dtype=int)
            self.cells6_out = np.array([], dtype=int)
            self.cells1_contain_out = np.array([], dtype=int)
            self.cells2_contain_out = np.array([], dtype=int)
            self.cells3_contain_out = np.array([], dtype=int)
            self.cells4_contain_out = np.array([], dtype=int)
            self.cells5_contain_out = np.array([], dtype=int)
            self.cells6_contain_out = np.array([], dtype=int)

    def diffusion_inward(self):
        """
        Ingoing diffusion from the outside.
        """
        # generating new particles on the diffusion surface (X = 0)
        new_in_page = np.random.randint(self.n_cells_per_axis, size=(2, self.number_inward_cells_pro_page))
        new_in_page = np.concatenate((new_in_page, np.zeros((1, self.number_inward_cells_pro_page), dtype=int)))
        # appending new generated particles as a ballistic ones to cells1
        new_in_page = np.array(np.ravel_multi_index(new_in_page, self.shape), dtype=int)
        self.cells1 = np.concatenate((self.cells1, new_in_page))

        # mixing particles according to Chopard and Droz model
        #  -------------------------------------------------------------------------------------------------------------
        # cell 1
        randomise = np.random.random_sample(len(self.cells1))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells1[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells1[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells1[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells1[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells1[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells1[temp_ind]))
        self.cells1 = np.array([], dtype=int)

        # cell 2
        randomise = np.random.random_sample(len(self.cells2))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells2[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells2[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells2[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells2[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells2[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells2[temp_ind]))
        self.cells2 = np.array([], dtype=int)

        # cell 3
        randomise = np.random.random_sample(len(self.cells3))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells3[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells3[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells3[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells3[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells3[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells3[temp_ind]))
        self.cells3 = np.array([], dtype=int)

        # cell 4
        randomise = np.random.random_sample(len(self.cells4))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells4[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells4[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells4[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells4[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells4[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells4[temp_ind]))
        self.cells4 = np.array([], dtype=int)

        # cell 5
        randomise = np.random.random_sample(len(self.cells5))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells5[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells5[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells5[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells5[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells5[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells5[temp_ind]))
        self.cells5 = np.array([], dtype=int)

        # cell 6
        randomise = np.random.random_sample(len(self.cells6))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range))
        self.cells1_contain = np.concatenate((self.cells1_contain, self.cells6[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range) & (randomise <= self.p2_range))
        self.cells2_contain = np.concatenate((self.cells2_contain, self.cells6[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range) & (randomise <= self.p3_range))
        self.cells3_contain = np.concatenate((self.cells3_contain, self.cells6[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range) & (randomise <= self.p4_range))
        self.cells5_contain = np.concatenate((self.cells5_contain, self.cells6[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range) & (randomise <= self.p_r_range))
        self.cells4_contain = np.concatenate((self.cells4_contain, self.cells6[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range)
        self.cells6_contain = np.concatenate((self.cells6_contain, self.cells6[temp_ind]))
        del temp_ind
        gc.collect()
        self.cells6 = np.array([], dtype=int)
        #  -------------------------------------------------------------------------------------------------------------

        # fixing in- and outgoing cells
        #  -------------------------------------------------------------------------------------------------------------
        # transform back to a non flat coordinates
        self.cells1_contain = np.array(np.unravel_index(self.cells1_contain, self.shape), dtype=int)
        self.cells2_contain = np.array(np.unravel_index(self.cells2_contain, self.shape), dtype=int)
        self.cells3_contain = np.array(np.unravel_index(self.cells3_contain, self.shape), dtype=int)
        self.cells4_contain = np.array(np.unravel_index(self.cells4_contain, self.shape), dtype=int)
        self.cells5_contain = np.array(np.unravel_index(self.cells5_contain, self.shape), dtype=int)
        self.cells6_contain = np.array(np.unravel_index(self.cells6_contain, self.shape), dtype=int)
        # adjusting a coordinates of side points for correct shifting
        self.cells1_contain = np.delete(self.cells1_contain,
                                        np.where(self.cells1_contain[2, :] == self.n_cells_per_axis - 1), 1)
        self.cells3_contain = np.delete(self.cells3_contain, np.where(self.cells3_contain[2, :] == 0), 1)
        self.cells2_contain[1, np.where(self.cells2_contain[1, :] == self.n_cells_per_axis - 1)] = -1
        self.cells4_contain[0, np.where(self.cells4_contain[0, :] == self.n_cells_per_axis - 1)] = -1
        self.cells5_contain[1, np.where(self.cells5_contain[1, :] == 0)] = self.n_cells_per_axis
        self.cells6_contain[0, np.where(self.cells6_contain[0, :] == 0)] = self.n_cells_per_axis
        # applying a shift, direction and periodic boundary conditions respectively
        self.cells1_contain[2, :] += 1
        self.cells2_contain[1, :] += 1
        self.cells3_contain[2, :] -= 1
        self.cells4_contain[0, :] += 1
        self.cells5_contain[1, :] -= 1
        self.cells6_contain[0, :] -= 1
        #  -------------------------------------------------------------------------------------------------------------

        # flatting and relocating particles back to initial cells + resetting temporary containers
        #  -------------------------------------------------------------------------------------------------------------
        self.cells1 = np.ravel_multi_index(self.cells1_contain, self.shape)
        self.cells1_contain = np.array([], dtype=int)
        self.cells2 = np.ravel_multi_index(self.cells2_contain, self.shape)
        self.cells2_contain = np.array([], dtype=int)
        self.cells3 = np.ravel_multi_index(self.cells3_contain, self.shape)
        self.cells3_contain = np.array([], dtype=int)
        self.cells4 = np.ravel_multi_index(self.cells4_contain, self.shape)
        self.cells4_contain = np.array([], dtype=int)
        self.cells5 = np.ravel_multi_index(self.cells5_contain, self.shape)
        self.cells5_contain = np.array([], dtype=int)
        self.cells6 = np.ravel_multi_index(self.cells6_contain, self.shape)
        self.cells6_contain = np.array([], dtype=int)
        #  -------------------------------------------------------------------------------------------------------------

    def diffusion_outward(self):
        """
        Outgoing diffusion from the inside.
        """
        # generating new particles on the surface (X = max)
        new_out_page = np.random.randint(self.n_cells_per_axis, size=(2, self.number_outward_cells_pro_page))
        new_out_page = np.concatenate((new_out_page, np.full((1, self.number_outward_cells_pro_page),
                                                             self.n_cells_per_axis - 1, dtype=int)))
        new_out_page = np.array(np.ravel_multi_index(new_out_page, self.shape), dtype=int)
        # appending new generated particles as a ballistic ones to cells1_out
        self.cells1_out = np.concatenate((self.cells1_out, new_out_page))

        # mixing particles according to Chopard and Droz model
        #  -------------------------------------------------------------------------------------------------------------
        # cell 1
        randomise = np.random.random_sample(len(self.cells1_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells1_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells1_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells1_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells1_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells1_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells1_out[temp_ind]))
        self.cells1_out = np.array([], dtype=int)

        # cell 2
        randomise = np.random.random_sample(len(self.cells2_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells2_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells2_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells2_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells2_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells2_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells2_out[temp_ind]))
        self.cells2_out = np.array([], dtype=int)

        # cell 3
        randomise = np.random.random_sample(len(self.cells3_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells3_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells3_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells3_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells3_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells3_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells3_out[temp_ind]))
        self.cells3_out = np.array([], dtype=int)

        # cell 4
        randomise = np.random.random_sample(len(self.cells4_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells4_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells4_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells4_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells4_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells4_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells4_out[temp_ind]))
        self.cells4_out = np.array([], dtype=int)

        # cell 5
        randomise = np.random.random_sample(len(self.cells5_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells5_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells5_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells5_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells5_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells5_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells5_out[temp_ind]))
        self.cells5_out = np.array([], dtype=int)

        # cell 6
        randomise = np.random.random_sample(len(self.cells6_out))
        # deflection 1
        temp_ind = np.where((randomise > 0) & (randomise <= self.p1_range_out))
        self.cells2_contain_out = np.concatenate((self.cells2_contain_out, self.cells6_out[temp_ind]))
        # deflection 2
        temp_ind = np.where((randomise > self.p1_range_out) & (randomise <= self.p2_range_out))
        self.cells1_contain_out = np.concatenate((self.cells1_contain_out, self.cells6_out[temp_ind]))
        # deflection 3
        temp_ind = np.where((randomise > self.p2_range_out) & (randomise <= self.p3_range_out))
        self.cells5_contain_out = np.concatenate((self.cells5_contain_out, self.cells6_out[temp_ind]))
        # deflection 4
        temp_ind = np.where((randomise > self.p3_range_out) & (randomise <= self.p4_range_out))
        self.cells3_contain_out = np.concatenate((self.cells3_contain_out, self.cells6_out[temp_ind]))
        # reflection
        temp_ind = np.where((randomise > self.p4_range_out) & (randomise <= self.p_r_range_out))
        self.cells4_contain_out = np.concatenate((self.cells4_contain_out, self.cells6_out[temp_ind]))
        # ballistic
        temp_ind = np.where(randomise > self.p_r_range_out)
        self.cells6_contain_out = np.concatenate((self.cells6_contain_out, self.cells6_out[temp_ind]))
        self.cells6_out = np.array([], dtype=int)
        del temp_ind
        gc.collect()
        #  -------------------------------------------------------------------------------------------------------------

        # fixing in- and outgoing cells
        #  -------------------------------------------------------------------------------------------------------------
        # transform back to a non flat coordinates
        self.cells1_contain_out = np.array(np.unravel_index(self.cells1_contain_out, self.shape), dtype=int)
        self.cells2_contain_out = np.array(np.unravel_index(self.cells2_contain_out, self.shape), dtype=int)
        self.cells3_contain_out = np.array(np.unravel_index(self.cells3_contain_out, self.shape), dtype=int)
        self.cells4_contain_out = np.array(np.unravel_index(self.cells4_contain_out, self.shape), dtype=int)
        self.cells5_contain_out = np.array(np.unravel_index(self.cells5_contain_out, self.shape), dtype=int)
        self.cells6_contain_out = np.array(np.unravel_index(self.cells6_contain_out, self.shape), dtype=int)
        # adjusting a coordinates of side points for correct shifting

        self.cells3_contain_out = np.delete(self.cells3_contain_out,
                                            np.where(self.cells3_contain_out[2, :] == self.n_cells_per_axis - 1), 1)
        self.cells1_contain_out = np.delete(self.cells1_contain_out, np.where(self.cells1_contain_out[2, :] == 0), 1)

        # self.cells1_contain_out[2, np.where(self.cells1_contain_out[2, :] == 0)] = self.n_cells_per_axis
        # self.cells3_contain_out[2, np.where(self.cells3_contain_out[2, :] == self.n_cells_per_axis - 1)] = -1

        self.cells2_contain_out[1, np.where(self.cells2_contain_out[1, :] == self.n_cells_per_axis - 1)] = -1
        self.cells4_contain_out[0, np.where(self.cells4_contain_out[0, :] == self.n_cells_per_axis - 1)] = -1
        self.cells5_contain_out[1, np.where(self.cells5_contain_out[1, :] == 0)] = self.n_cells_per_axis
        self.cells6_contain_out[0, np.where(self.cells6_contain_out[0, :] == 0)] = self.n_cells_per_axis
        # applying a shift, direction and periodic boundary conditions respectively
        self.cells1_contain_out[2, :] -= 1
        self.cells2_contain_out[1, :] += 1
        self.cells3_contain_out[2, :] += 1
        self.cells4_contain_out[0, :] += 1
        self.cells5_contain_out[1, :] -= 1
        self.cells6_contain_out[0, :] -= 1
        #  -------------------------------------------------------------------------------------------------------------

        # relocating particles back to initial cells and resetting temporary containers
        #  -------------------------------------------------------------------------------------------------------------
        self.cells1_out = self.cells1_contain_out
        self.cells1_contain_out = np.array([], dtype=int)
        self.cells2_out = self.cells2_contain_out
        self.cells2_contain_out = np.array([], dtype=int)
        self.cells3_out = self.cells3_contain_out
        self.cells3_contain_out = np.array([], dtype=int)
        self.cells4_out = self.cells4_contain_out
        self.cells4_contain_out = np.array([], dtype=int)
        self.cells5_out = self.cells5_contain_out
        self.cells5_contain_out = np.array([], dtype=int)
        self.cells6_out = self.cells6_contain_out
        self.cells6_contain_out = np.array([], dtype=int)
        #  -------------------------------------------------------------------------------------------------------------

        # flatten coordinates
        #  -------------------------------------------------------------------------------------------------------------
        if not self.param["compute_precipitations"]:
            self.cells1_out = np.ravel_multi_index(self.cells1_out, self.shape)
            self.cells2_out = np.ravel_multi_index(self.cells2_out, self.shape)
            self.cells3_out = np.ravel_multi_index(self.cells3_out, self.shape)
            self.cells4_out = np.ravel_multi_index(self.cells4_out, self.shape)
            self.cells5_out = np.ravel_multi_index(self.cells5_out, self.shape)
            self.cells6_out = np.ravel_multi_index(self.cells6_out, self.shape)
        #  -------------------------------------------------------------------------------------------------------------

    def save_results(self, iteration):
        if self.param["inward_diffusion"]:
            save_data = np.concatenate((self.cells1, self.cells2, self.cells3,
                                        self.cells4, self.cells5, self.cells6))
            self.utils.db.insert_inward(save_data, iteration)

        if self.param["outward_diffusion"]:
            save_data = np.concatenate((self.cells1_out, self.cells2_out, self.cells3_out,
                                        self.cells4_out, self.cells5_out, self.cells6_out))
            self.utils.db.insert_outward(save_data, iteration)
        del save_data
        gc.collect()
        self.utils.db.insert_lasti(iteration)

    def simulation(self):
        """
        Simulation of diffusion and precipitation formation.
        """
        for iteration in progressbar.progressbar(range(self.n_iterations)):
            # performing a both diffusions

            p1 = multiprocessing.Process(target=self.diffusion_inward)
            p2 = multiprocessing.Process(target=self.diffusion_outward)

            p1.start()
            p2.start()

            p1.join()
            p2.join()

            # self.utils.db.insert_lasti(iteration)

            # save data
            # if self.save_whole and iteration != self.n_iterations - 1:
            #     self.save_results(iteration)

        # self.save_results(self.n_iterations - 1)
        # end = time.time()
        # self.elapsed_time = (end - self.begin)
        # self.utils.db.insert_time(self.elapsed_time)
        # self.utils.db.conn.commit()


if __name__ == '__main__':
    param = {"matrix_elem": "Ni",
             "cond_inward": "Test",
             "cond_outward": "Test",
             "diff_in_precipitation": 6.18 * 10 ** -16,
             "temperature": 1100,
             "n_cells_per_axis": 102,  # ONLY MULTIPLES OF 3 ARE ALLOWED!!!!!!!
             "n_iterations": 1000,  # must be >= n_cells_per_axis!!!!!!!!!!!
             "sim_time": 3600,
             "size": 0.001,

             "diff_elem_conc": 0.01,
             "active_elem_conc": 0.05,
             "active_elem_conc_real": 0.02,
             "threshold_inward": 2,
             "threshold_outward": 1,
             "threshold_growth_inward": 0,
             "threshold_growth_outward": 1,
             "sol_prod": 1.4 * 10**-6,

             "inward_diffusion": True,
             "outward_diffusion": True,
             "compute_precipitations": False,
             "grow_precipitations": False,
             "consider_diffusion_in_precipitation": False,
             "save_whole": True,
             "save_precipitation_front": False}

    eng = CellularAutomata(param=param)
    eng.simulation()
