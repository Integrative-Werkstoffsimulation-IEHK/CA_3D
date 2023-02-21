from utilities import *
import gc
import progressbar


class CADiffVal:
    """

    """
    def __init__(self, half_thickness, test_type="DS", geometry="cube", periodic_bc=False):
        # pre settings
        self.param = physical_data.DEFAULT_PARAM_VAL
        if test_type == "DS":
            self.param["outward_diffusion"] = True
            self.param["inward_diffusion"] = False
        elif test_type == "SS":
            self.param["outward_diffusion"] = False
            self.param["inward_diffusion"] = True
        else:
            self.brake = True
        self.utils = Utils(self.param)
        self.save_whole = self.param["save_whole"]
        self.begin = time.time()
        self.elapsed_time = 0
        self.periodic_bc = periodic_bc
        self.brake = False

        # simulated space parameters
        self.n_cells_per_axis = self.param["n_cells_per_axis"]
        self.half_thickness = half_thickness
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

        # setting variables for outward diffusion
        if self.param["outward_diffusion"]:
            self.p_out = self.utils.p_out
            self.p1_range_out = 1 * self.p_out["p"]
            self.p2_range_out = 2 * self.p_out["p"]
            self.p3_range_out = 3 * self.p_out["p"]
            self.p4_range_out = 4 * self.p_out["p"]
            self.p_r_range_out = self.p4_range_out + self.p_out["p3"]

            middle = int(self.n_cells_per_axis / 2)
            minus = middle - self.half_thickness
            plus = middle + self.half_thickness

            if geometry == "wall":
                # initial geometry -> WALL
                self.cells1_out = np.zeros(self.shape, dtype=int)
                self.cells1_out[:, :, minus:middle] = 1
                self.cells1_out = np.nonzero(self.cells1_out)
                self.cells1_out = np.ravel_multi_index(self.cells1_out, self.shape)

                self.cells3_out = np.zeros(self.shape, dtype=int)
                self.cells3_out[:, :, middle:plus] = 1
                self.cells3_out = np.nonzero(self.cells3_out)
                self.cells3_out = np.ravel_multi_index(self.cells3_out, self.shape)

                self.cells2_out = np.array([], dtype=int)
                self.cells4_out = np.array([], dtype=int)
                self.cells5_out = np.array([], dtype=int)
                self.cells6_out = np.array([], dtype=int)

            elif geometry == "column":
                # initial geometry -> COLUMN
                self.cells1_out = np.zeros(self.shape, dtype=int)
                self.cells1_out[:, minus:plus, minus:middle] = 1
                self.cells1_out = np.nonzero(self.cells1_out)
                self.cells1_out = np.ravel_multi_index(self.cells1_out, self.shape)

                self.cells3_out = np.zeros(self.shape, dtype=int)
                self.cells3_out[:, minus:plus, middle:plus] = 1
                self.cells3_out = np.nonzero(self.cells3_out)
                self.cells3_out = np.ravel_multi_index(self.cells3_out, self.shape)

                self.cells2_out = np.array([], dtype=int)
                self.cells4_out = np.array([], dtype=int)
                self.cells5_out = np.array([], dtype=int)
                self.cells6_out = np.array([], dtype=int)

            elif geometry == "cube":
                # initial geometry -> CUBE
                self.cells1_out = np.zeros(self.shape, dtype=int)
                self.cells1_out[minus:plus, minus:plus, minus:middle] = 1
                self.cells1_out = np.nonzero(self.cells1_out)
                self.cells1_out = np.ravel_multi_index(self.cells1_out, self.shape)

                self.cells3_out = np.zeros(self.shape, dtype=int)
                self.cells3_out[minus:plus, minus:plus, middle:plus] = 1
                self.cells3_out = np.nonzero(self.cells3_out)
                self.cells3_out = np.ravel_multi_index(self.cells3_out, self.shape)

                self.cells2_out = np.zeros(self.shape, dtype=int)
                self.cells2_out[minus:plus, middle:plus, minus:plus] = 1
                self.cells2_out = np.nonzero(self.cells2_out)
                self.cells2_out = np.ravel_multi_index(self.cells2_out, self.shape)

                self.cells4_out = np.zeros(self.shape, dtype=int)
                self.cells4_out[middle:plus, minus:plus, minus:plus] = 1
                self.cells4_out = np.nonzero(self.cells4_out)
                self.cells4_out = np.ravel_multi_index(self.cells4_out, self.shape)

                self.cells5_out = np.zeros(self.shape, dtype=int)
                self.cells5_out[minus:plus, minus:middle, minus:plus] = 1
                self.cells5_out = np.nonzero(self.cells5_out)
                self.cells5_out = np.ravel_multi_index(self.cells5_out, self.shape)

                self.cells6_out = np.zeros(self.shape, dtype=int)
                self.cells6_out[minus:middle, minus:plus, minus:plus] = 1
                self.cells6_out = np.nonzero(self.cells6_out)
                self.cells6_out = np.ravel_multi_index(self.cells6_out, self.shape)

            elif geometry == "point" and self.n_cells_per_axis % 2 != 0:
                self.half_thickness = 1
                # initial geometry -> POINT
                central_point = [int(self.n_cells_per_axis/2), int(self.n_cells_per_axis/2), int(self.n_cells_per_axis/2)]
                central_point = np.ravel_multi_index(central_point, self.shape)
                self.cells1_out = np.ones(50, dtype=int) * central_point
                self.cells2_out = np.ones(50, dtype=int) * central_point
                self.cells3_out = np.ones(50, dtype=int) * central_point
                self.cells4_out = np.ones(50, dtype=int) * central_point
                self.cells5_out = np.ones(50, dtype=int) * central_point
                self.cells6_out = np.ones(50, dtype=int) * central_point
            else:
                self.brake = True
            self.cells1_contain_out = np.array([], dtype=int)
            self.cells2_contain_out = np.array([], dtype=int)
            self.cells3_contain_out = np.array([], dtype=int)
            self.cells4_contain_out = np.array([], dtype=int)
            self.cells5_contain_out = np.array([], dtype=int)
            self.cells6_contain_out = np.array([], dtype=int)

            if not self.brake:
                items = np.concatenate((self.cells1_out, self.cells2_out, self.cells3_out,
                                        self.cells4_out, self.cells5_out, self.cells6_out))
                inward = np.zeros(self.n_cells_per_axis, dtype=int)
                precipitations = np.zeros(self.n_cells_per_axis, dtype=int)
                outward = np.array(np.unravel_index(items, self.shape))
                outward = [len(np.where(outward[2] == i)[0]) for i in range(self.n_cells_per_axis)]
                n_matrix_page = self.n_cells_per_axis ** 2
                matrix_n = [n_matrix_page - n_pre for n_pre in precipitations]

                self.m_inward = self.utils.masses["inward"]
                self.m_matrix = self.utils.masses["matrix"]
                self.m_act = self.utils.masses["active"]
                self.m_precip = self.utils.masses["precipitation"]

                inward = [n_in * self.m_inward * 100 /
                          (n_in * self.m_inward + mat * self.m_matrix + n_out * self.m_act + n_pre *
                           self.m_precip) for n_in, n_out, n_pre, mat in
                          zip(inward, outward, precipitations, matrix_n)]

                precipitations = [n_pre * self.m_precip * 100 /
                                  (n_in * self.m_inward + mat * self.m_matrix + n_out * self.m_act
                                   + n_pre * self.m_precip) for n_in, n_out, n_pre, mat in
                                  zip(inward, outward, precipitations, matrix_n)]

                outward = [n_out * self.m_act * 100 /
                           (n_in * self.m_inward + mat * self.m_matrix + n_out * self.m_act + n_pre *
                            self.m_precip) for n_in, n_out, n_pre, mat in
                           zip(inward, outward, precipitations, matrix_n)]
                self.utils.db.insert_difval_info(max(outward), self.half_thickness)

    def diffusion_single_side(self):
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

    def diffusion_double_side(self):
        """
        Outgoing diffusion from the inside.
        """
        # generating new particles on the surface (X = max)
        # new_out_page = np.random.randint(self.n_cells_per_axis, size=(2, self.number_outward_cells_pro_page))
        # new_out_page = np.concatenate((new_out_page, np.full((1, self.number_outward_cells_pro_page),
        #                                                      self.n_cells_per_axis - 1, dtype=int)))
        # new_out_page = np.array(np.ravel_multi_index(new_out_page, self.shape), dtype=int)
        # # appending new generated particles as a ballistic ones to cells1_out
        # self.cells1_out = np.concatenate((self.cells1_out, new_out_page))

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

        if self.periodic_bc:
            self.cells1_contain_out[2, np.where(self.cells1_contain_out[2, :] == 0)] = self.n_cells_per_axis
            self.cells3_contain_out[2, np.where(self.cells3_contain_out[2, :] == self.n_cells_per_axis - 1)] = -1
            self.cells2_contain_out[1, np.where(self.cells2_contain_out[1, :] == self.n_cells_per_axis - 1)] = -1
            self.cells4_contain_out[0, np.where(self.cells4_contain_out[0, :] == self.n_cells_per_axis - 1)] = -1
            self.cells5_contain_out[1, np.where(self.cells5_contain_out[1, :] == 0)] = self.n_cells_per_axis
            self.cells6_contain_out[0, np.where(self.cells6_contain_out[0, :] == 0)] = self.n_cells_per_axis
        else:
            self.cells3_contain_out = np.delete(self.cells3_contain_out,
                                                np.where(self.cells3_contain_out[2, :] == self.n_cells_per_axis - 1), 1)
            self.cells1_contain_out = np.delete(self.cells1_contain_out,
                                                np.where(self.cells1_contain_out[2, :] == 0), 1)
            self.cells2_contain_out = np.delete(self.cells2_contain_out,
                                                np.where(self.cells2_contain_out[1, :] == self.n_cells_per_axis - 1), 1)
            self.cells4_contain_out = np.delete(self.cells4_contain_out,
                                                np.where(self.cells4_contain_out[0, :] == self.n_cells_per_axis - 1), 1)
            self.cells5_contain_out = np.delete(self.cells5_contain_out,
                                                np.where(self.cells5_contain_out[1, :] == 0), 1)
            self.cells6_contain_out = np.delete(self.cells6_contain_out,
                                                np.where(self.cells6_contain_out[0, :] == 0), 1)

        # applying a shift, direction respectively
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
            self.utils.db.insert_particle_data("inward", iteration, save_data)
        if self.param["outward_diffusion"]:
            save_data = np.concatenate((self.cells1_out, self.cells2_out, self.cells3_out,
                                        self.cells4_out, self.cells5_out, self.cells6_out))
            self.utils.db.insert_particle_data("outward", iteration, save_data)
        self.utils.db.insert_lasti(iteration)

    def simulation(self):
        """
        Simulation of diffusion and precipitation formation.
        """
        if not self.brake:
            for iteration in progressbar.progressbar(range(self.n_iterations)):
                # performing a both diffusions
                if self.param["inward_diffusion"]:
                    self.diffusion_single_side()

                if self.param["outward_diffusion"]:
                    self.diffusion_double_side()

                self.utils.db.insert_lasti(iteration)

                # save data
                if self.save_whole and iteration != self.n_iterations - 1:
                    self.save_results(iteration)

            self.save_results(self.n_iterations - 1)
            end = time.time()
            self.elapsed_time = (end - self.begin)
            self.utils.db.insert_time(self.elapsed_time)
            self.utils.db.conn.commit()
        else:
            return print("WRONG NUMBER OF CELLS PER AXIS FOR CHOSEN INITIAL GEOMETRY or WRONG TEST TYPE!!!")


if __name__ == '__main__':
    en = CADiffVal(10, test_type="DS", geometry="cube")
    en.simulation()
