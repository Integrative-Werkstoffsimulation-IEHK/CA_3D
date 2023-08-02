import numpy as np
from utils.my_data_structs import *
from utils.utilities import *
from utils.numba_functions import *
import progressbar
import elements
import time
from thermodynamics import td_data


class CellularAutomata:
    """
    TODO: 1 -> Write comments.
          2 -> Check aggregated indexes!
    """

    def __init__(self, user_input=None):
        # pre settings
        if user_input is None:
            # setting default parameters if no user input is given
            user_input = templates.DEFAULT_PARAM
        self.utils = Utils(user_input)
        self.utils.create_database()
        self.utils.generate_param()
        self.utils.print_init_var()
        self.param = self.utils.param
        self.elapsed_time = 0

        # simulated space parameters
        self.cells_per_axis = self.param["n_cells_per_axis"]
        self.cells_per_page = self.cells_per_axis ** 2
        self.extended_axis = self.cells_per_axis + self.param["neigh_range"]
        self.matrix_mass = np.full(self.cells_per_axis, self.cells_per_page) * self.param["matrix_elem"]["mass_per_cell"]
        self.shape = (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis)
        self.extended_shape = (self.cells_per_axis, self.cells_per_axis, self.extended_axis)
        self.n_iter = self.param["n_iterations"]
        self.iteration = None
        self.curr_max_furthest = 0
        self.objs = templates.DEFAULT_OBJ_REF

        # setting objects for inward diffusion
        if self.param["inward_diffusion"]:
            self.primary_oxidant = elements.OxidantElem(self.param["oxidant"]["primary"], self.utils)
            self.objs[0]["oxidant"] = self.primary_oxidant
            self.objs[1]["oxidant"] = self.primary_oxidant
            self.primary_oxidant.diffuse = self.primary_oxidant.diffuse_with_scale
            # self.primary_oxidant.diffuse = self.primary_oxidant.diffuse_bulk

            if self.param["secondary_oxidant_exists"]:
                self.secondary_oxidant = elements.OxidantElem(self.param["oxidant"]["secondary"], self.utils)
                self.objs[2]["oxidant"] = self.primary_oxidant
                self.objs[3]["oxidant"] = self.primary_oxidant

        # setting objects for outward diffusion
        if self.param["outward_diffusion"]:
            self.primary_active = elements.ActiveElem(self.param["active_element"]["primary"])
            self.objs[0]["active"] = self.primary_active
            self.objs[2]["active"] = self.primary_active
            self.primary_active.diffuse = self.primary_active.diffuse_with_scale
            # self.primary_active.diffuse = self.primary_active.diffuse_bulk

            if self.param["secondary_active_element_exists"]:
                self.secondary_active = elements.ActiveElem(self.param["active_element"]["secondary"])
                self.objs[1]["active"] = self.secondary_active
                self.objs[3]["active"] = self.secondary_active

        # setting objects for precipitations
        if self.param["compute_precipitations"]:
            self.coord_buffer = MyBufferCoords(self.param["product"]["primary"]["oxidation_number"] *
                                           self.cells_per_axis ** 3)
            self.to_dissol_pn_buffer = MyBufferCoords(self.param["product"]["primary"]["oxidation_number"] *
                                           self.cells_per_axis ** 3)
            # self.cumul_product = np.full(self.shape, 0, dtype=np.ubyte)
            self.case = 0

            self.primary_product = elements.Product(self.param["product"]["primary"])
            self.objs[0]["product"] = self.primary_product
            if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                self.secondary_product = elements.Product(self.param["product"]["secondary"])
                self.objs[1]["product"] = self.secondary_product
                self.ternary_product = elements.Product(self.param["product"]["ternary"])
                self.objs[2]["product"] = self.ternary_product
                self.quaternary_product = elements.Product(self.param["product"]["quaternary"])
                self.objs[3]["product"] = self.quaternary_product
                self.objs[0]["to_check_with"] = self.cumul_product
                self.objs[1]["to_check_with"] = self.cumul_product
                self.objs[2]["to_check_with"] = self.cumul_product
                self.objs[3]["to_check_with"] = self.cumul_product

                self.precip_func = self.precipitation_2_cells
                self.calc_precip_front = self.calc_precip_front_2
                # self.decomposition = self.decomposition_0

            elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                self.secondary_product = elements.Product(self.param["product"]["secondary"])
                self.objs[1]["product"] = self.secondary_product
                self.objs[0]["to_check_with"] = self.secondary_product
                self.objs[1]["to_check_with"] = self.primary_product
                self.precip_func = self.precipitation_1_cells
                self.calc_precip_front = self.calc_precip_front_1
                # self.decomposition = self.decomposition_0

            else:
                self.precip_func = self.precipitation_0_cells
                # self.precip_func = self.precipitation_0
                self.calc_precip_front = self.calc_precip_front_0
                # self.decomposition = self.decomposition_0
                self.primary_oxidant.scale = self.primary_product
                self.primary_active.scale = self.primary_product

                self.primary_oxid_numb = self.param["product"]["primary"]["oxidation_number"]
                if self.primary_oxid_numb == 1:
                    self.go_around = self.go_around_single_oxid_n
                    self.fix_init_precip = self.fix_init_precip_bool
                    self.precipitations3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1),
                        False, dtype=bool)
                else:
                    self.go_around = self.go_around_mult_oxid_n
                    self.fix_init_precip = self.fix_init_precip_int
                    self.precipitations3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1),
                        False, dtype=np.ubyte)

            # self.precipitations3d = np.full(self.shape, False)
            # self.half_thickness = 20
            # middle = int(self.cells_per_axis / 2)
            # minus = middle - self.half_thickness
            # plus = middle + self.half_thickness
            # self.primary_product.c3d[minus:plus, minus:plus, minus:plus] = 4

            self.primary_product.c3d[1, 1, 50] = 4
            self.primary_product.full_c3d[1, 1, 50] = True

            self.primary_product.c3d[1, 1, 49] = 4
            self.primary_product.full_c3d[1, 1, 49] = True

            self.primary_product.c3d[1, 2, 49] = 4
            self.primary_product.full_c3d[1, 2, 49] = True

            self.primary_product.c3d[1, 2, 50] = 4
            self.primary_product.full_c3d[1, 2, 50] = True


            self.primary_product.c3d[2, 1, 50] = 4
            self.primary_product.full_c3d[2, 1, 50] = True

            self.primary_product.c3d[2, 1, 49] = 4
            self.primary_product.full_c3d[2, 1, 49] = True

            self.primary_product.c3d[2, 2, 49] = 4
            self.primary_product.full_c3d[2, 2, 49] = True

            self.primary_product.c3d[2, 2, 50] = 4
            self.primary_product.full_c3d[2, 2, 50] = True


            self.primary_product.c3d[2, 0, 50] = 4
            self.primary_product.full_c3d[2, 0, 50] = True

            self.primary_product.c3d[2, 1, 51] = 4
            self.primary_product.full_c3d[2, 1, 51] = True

            self.primary_product.c3d[3, 1, 50] = 4
            self.primary_product.full_c3d[3, 1, 50] = True


            self.primary_product.c3d[3, 3, 50] = 2
            self.primary_product.c3d[10, 3, 50] = 2
            self.primary_product.c3d[35, 3, 50] = 2

            self.primary_product.c3d[45, 3, 50] = 4
            self.primary_product.full_c3d[45, 3, 50] = True

            # self.primary_product.full_c3d[minus:plus, minus:plus, minus:plus] = True
            # self.primary_product.full_c3d[1, 1, 50] = True
            # shift = 0
            # self.precipitations3d[minus + shift:plus + shift, minus + shift:plus + shift, minus + shift:plus + shift] = True
            # self.precipitations = np.array(np.nonzero(self.precipitations), dtype=int)
            # self.precipitations3d = np.full(self.single_page_shape, False)
            # self.precipitations3d_sec = np.full(self.single_page_shape, False)

            self.threshold_inward = self.param["threshold_inward"]
            if self.threshold_inward < 2:
                self.check_intersection = self.ci_single
            else:
                self.check_intersection = self.ci_multi
            self.threshold_outward = self.param["threshold_outward"]

            self.fetch_ind = None
            self.generate_fetch_ind()

            self.dissol_pn = self.param["dissolution_p"] ** (1 / self.param["dissolution_n"])
            power = (self.param["dissolution_n"] - 1)/self.param["dissolution_n"]
            self.const_b_dissol = (1/3) * log(self.param["block_scale_factor"] * (self.param["dissolution_p"] ** power))
            self.p_b_3 = self.dissol_pn * 2.718281828 ** (self.const_b_dissol * 3) / (self.param["block_scale_factor"])
            self.p_b_2 = self.dissol_pn * 2.718281828 ** (self.const_b_dissol * 4) / (self.param["block_scale_factor"]**3)
            self.p_b_1 = self.dissol_pn * 2.718281828 ** (self.const_b_dissol * 5) / (self.param["block_scale_factor"]**10)

            self.aggregated_ind = [[7, 0, 1, 2, 19, 16, 14],
                                   [6, 0, 1, 5, 18, 15, 14],
                                   [8, 0, 4, 5, 20, 15, 17],
                                   [9, 0, 4, 2, 21, 16, 17],
                                   [11, 3, 1, 2, 19, 24, 22],
                                   [10, 3, 1, 5, 18, 23, 22],
                                   [12, 3, 4, 5, 20, 23, 25],
                                   [13, 3, 4, 2, 21, 24, 25]]

            self.nucl_prob = probabilities.NucleationProbabilities(self.param)
            self.dissol_prob = probabilities.DissolutionProbabilities(self.param)

            self.furthest_index = 0
            self.comb_indexes = None
            self.rel_prod_fraction = None
            self.product_indexes = None

            self.max_inside_neigh_number = 6 * self.primary_oxid_numb
            self.max_block_neigh_number = 7 * self.primary_oxid_numb

            # self.look_up_table = td_data.TDATA()

        self.begin = time.time()

    def simulation(self):
        """
        """
        for self.iteration in progressbar.progressbar(range(self.n_iter)):
            if self.param["compute_precipitations"]:
                self.precip_func()
                self.dissolution_0_cells()
            # if self.param["decompose_precip"]:
            #     self.decomposition_0()
            if self.param["inward_diffusion"]:
                self.diffusion_inward()
            if self.param["outward_diffusion"]:
                self.diffusion_outward()
            if self.param["save_whole"]:
                self.save_results_only_prod()
            # if self.iteration > 50000:
            #     break

        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.utils.db.insert_time(self.elapsed_time)
        self.utils.db.conn.commit()

    def decomposition_0(self, plane_indxs):
        # self.precipitations = self.primary_product.transform_c3d()
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, plane_indxs]))
        self.precipitations = nz_ind
        self.precipitations[2] = plane_indxs[nz_ind[2]]

        if len(self.precipitations[0]) > 0:
            dec_p_three_open = np.array([[], [], []], dtype=np.short)
            dec_p_two_open = np.array([[], [], []], dtype=np.short)
            dec_p_one_open = np.array([[], [], []], dtype=np.short)

            all_arounds = self.utils.calc_sur_ind_decompose(self.precipitations)

            neighbours = go_around_bool(self.primary_product.c3d, all_arounds)
            arr_len_flat = np.array([np.sum(item[:6]) for item in neighbours], dtype=np.ubyte)
            arr_len_corners = np.array([np.sum(item[6:14]) for item in neighbours], dtype=np.ubyte)
            arr_len_side_corners = np.array([np.sum(item[14:]) for item in neighbours], dtype=np.ubyte)
            arr_len_all = arr_len_flat + arr_len_corners + arr_len_side_corners
            # These are inside!
            # ------------------------------------------------
            index = np.where(arr_len_flat < 6)[0]
            if len(index) < len(self.precipitations[0]):
                self.precipitations = self.precipitations[:, index]
                arr_len_flat = arr_len_flat[index]
                arr_len_corners = arr_len_corners[index]
                arr_len_side_corners = arr_len_side_corners[index]
                arr_len_all = arr_len_all[index]
                neighbours = neighbours[index]
            # ------------------------------------------------
            # These are not in a block => pn!
            # ------------------------------------------------
            index = np.where(arr_len_all >= 7)[0]
            dec_pn = np.delete(self.precipitations, index, 1)
            arr_len_flat_to_pn = np.delete(arr_len_flat, index)
            if len(index) > 0:
                self.precipitations = self.precipitations[:, index]
                # arr_len_flat_to_pn = np.delete(arr_len_flat, index)
                arr_len_flat = arr_len_flat[index]
                arr_len_corners = arr_len_corners[index]
                arr_len_side_corners = arr_len_side_corners[index]
                neighbours = neighbours[index]
                # ------------------------------------------------
                # These are not in a block => pn!
                # ------------------------------------------------
                index = np.where((arr_len_flat > 2) & (arr_len_corners > 0) & (arr_len_side_corners > 2))[0]
                if len(index) > 0:
                    dec_pn = np.concatenate((dec_pn, np.delete(self.precipitations, index, 1)), axis=1)
                    arr_len_flat_to_pn = np.concatenate((arr_len_flat_to_pn, np.delete(arr_len_flat, index)))
                    self.precipitations = self.precipitations[:, index]
                    arr_len_flat = arr_len_flat[index]
                    neighbours = neighbours[index]
                # ------------------------------------------------

                if np.any(self.precipitations):
                    aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in neighbours],
                                           dtype=int)
                    # These are not in a block => pn!
                    # ------------------------------------------------
                    where_blocks = np.unique(np.where(aggregation == 7)[0])
                    dec_pn = np.concatenate((dec_pn, np.delete(self.precipitations, where_blocks, 1)), axis=1)
                    arr_len_flat_to_pn = np.concatenate((arr_len_flat_to_pn, np.delete(arr_len_flat, where_blocks)))
                    # ------------------------------------------------
                    # These are in a block => p!
                    # ------------------------------------------------
                    self.precipitations = self.precipitations[:, where_blocks]
                    arr_len_flat = arr_len_flat[where_blocks]
                    one_open = np.where(arr_len_flat == 5)[0]
                    two_open = np.where(arr_len_flat == 4)[0]
                    three_open = np.where(arr_len_flat == 3)[0]

                    dec_p_three_open = self.precipitations[:, three_open]
                    dec_p_two_open = self.precipitations[:, two_open]
                    dec_p_one_open = self.precipitations[:, one_open]
                    # ------------------------------------------------

            self.precipitations = np.array([[], [], []], dtype=np.short)
            # needed_prob = self.dissol_pn * 2.718281828 ** (-self.param["exponent_power"] * arr_len_flat_to_pn)
            needed_prob = self.dissol_pn * 2.718281828 ** (self.const_b_dissol * arr_len_flat_to_pn)
            randomise = np.random.random_sample(len(arr_len_flat_to_pn))
            temp_ind = np.where(randomise < needed_prob)[0]
            dec_pn = dec_pn[:, temp_ind]

            randomise = np.random.random_sample(len(dec_p_three_open[0]))
            temp_ind = np.where(randomise < self.p_b_3)[0]
            dec_p_three_open = dec_p_three_open[:, temp_ind]

            randomise = np.random.random_sample(len(dec_p_two_open[0]))
            temp_ind = np.where(randomise < self.p_b_2)[0]
            dec_p_two_open = dec_p_two_open[:, temp_ind]

            randomise = np.random.random_sample(len(dec_p_one_open[0]))
            temp_ind = np.where(randomise < self.p_b_1)[0]
            dec_p_one_open = dec_p_one_open[:, temp_ind]

            todec = np.concatenate((dec_pn, dec_p_three_open, dec_p_two_open, dec_p_one_open), axis=1)

            if len(todec[0]) > 0:
                counts = np.unique(np.ravel_multi_index(todec, self.shape), return_counts=True)
                dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short)
                counts = np.array(counts[1], dtype=np.ubyte)

                self.primary_product.c3d[dec[0], dec[1], dec[2]] -= counts
                self.primary_product.full_c3d[dec[0], dec[1], dec[2]] = False
                self.primary_active.c3d[dec[0], dec[1], dec[2]] += counts

                # push back the oxidant cells to avoid nucl + disol jumps
                # to_shift_back = np.where(todec[2] > 0)
                # todec[2] -= 1
                # todec[2, to_shift_back] -= 1

                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, todec), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(todec[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def decomposition_original(self, plane_indxs):
        # plane_indxs = np.array([50])
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, plane_indxs]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(plane_indxs[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            full_ind = check_at_coord_dissol(self.primary_product.full_c3d, self.coord_buffer.get_buffer())

            if len(full_ind) > 0:
                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(full_ind))
                # precips = self.coord_buffer.get_elem_at_ind(full_ind)

                all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_elem_at_ind(full_ind))
                all_neigh = go_around_bool_dissol(self.primary_product.full_c3d, all_arounds)
                # int_all_neigh = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                #                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                #                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 1],
                #                           [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4],
                #                           [4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]])
                #
                # where_full = np.where(int_all_neigh == self.primary_oxid_numb)
                # boo_all_neigh = np.full(int_all_neigh.shape, False, dtype=bool)
                # boo_all_neigh[where_full] = True

                arr_len_flat = np.array([np.sum(item[:6]) for item in all_neigh], dtype=np.ubyte)
                # arr_len_corners = np.array([np.sum(item[6:14]) for item in all_neigh], dtype=np.ubyte)
                # arr_len_side_corners = np.array([np.sum(item[14:]) for item in all_neigh], dtype=np.ubyte)
                # arr_len_all = arr_len_flat + arr_len_corners + arr_len_side_corners

                index_outside = np.where((arr_len_flat < 6))[0]
                # index_inside = np.delete(full_ind, index_outside)
                # self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(full_ind[index_inside]))

                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(full_ind[index_outside]))
                # dissolve_pn_ind = self.coord_buffer.get_elem_instead_ind(full_ind[index_inside])
                # ind_pos_bl = np.where((2 < arr_len_flat[index_outside]) & (arr_len_all[index_outside] >= 7) &
                #                  (arr_len_corners[index_outside] > 0) & (arr_len_side_corners[index_outside] > 2))[0]

                aggregation = np.array(
                            [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh[index_outside]],
                            dtype=np.ubyte)
                ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                else:
                    # self.to_dissol_pn_buffer.append_to_buffer(
                    #     self.coord_buffer.get_elem_instead_ind(full_ind[index_inside]))
                    self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_buffer())
                    self.coord_buffer.reset_buffer()
            else:
                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(full_ind))
                self.coord_buffer.reset_buffer()

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.dissol_prob.dissol_prob_pp[to_dissolve_pn[2]]

            to_dissolve_p = self.coord_buffer.get_buffer()
            probs_p = self.dissol_prob.dissol_prob_pp[to_dissolve_p[2]] ** self.param["dissolution_n"]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_p[0]))
            temp_ind = np.where(randomise < probs_p)[0]
            to_dissolve_p = to_dissolve_p[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissolve_p), axis=1)

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]

                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts

                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def decomposition_adapted(self, plane_indxs):
        # plane_indxs = np.array([50])
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, plane_indxs]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(plane_indxs[nz_ind[2]], axis=2)

        all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_buffer())
        all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

        all_neigh_pn = all_neigh[[]]
        all_neigh_block = all_neigh[[]]

        # choose all the coordinates which have at least one full side neighbour
        where_full = np.unique(np.where(all_neigh[:, :6].view() == self.primary_oxid_numb)[0])

        to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
        self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))
        # self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(where_full))

        if self.coord_buffer.last_in_buffer > 0:
            all_neigh = all_neigh[where_full]

            arr_len_flat = np.array([np.sum(item[:6]) for item in all_neigh], dtype=np.ubyte)
            index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
            all_neigh = all_neigh[index_outside]

            aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh],
                                   dtype=np.ubyte)
            ind_where_blocks = np.unique(np.where(aggregation == self.max_block_neigh_number)[0])

            if len(ind_where_blocks) > 0:
                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)

                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                all_neigh_block = all_neigh[ind_where_blocks]
            else:
                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                all_neigh_pn = all_neigh

                self.coord_buffer.reset_buffer()
                all_neigh_block = all_neigh[[]]

        probs_pn_no_neigh = self.dissol_prob.dissol_prob_pp[to_dissol_pn_no_neigh[2]]

        to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
        all_neigh_pn = np.array([np.sum(item[:6]) for item in all_neigh_pn])
        probs_pn = self.dissol_prob.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

        to_dissolve_p = self.coord_buffer.get_buffer()
        all_neigh_block = np.array([np.sum(item[:6]) for item in all_neigh_block])
        probs_p = self.dissol_prob.get_probabilities_block(all_neigh_block, to_dissolve_p[2])

        randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
        temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
        to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

        randomise = np.random.random_sample(len(to_dissolve_pn[0]))
        temp_ind = np.where(randomise < probs_pn)[0]
        to_dissolve_pn = to_dissolve_pn[:, temp_ind]

        randomise = np.random.random_sample(len(to_dissolve_p[0]))
        temp_ind = np.where(randomise < probs_p)[0]
        to_dissolve_p = to_dissolve_p[:, temp_ind]

        to_dissolve = np.concatenate((to_dissolve_pn, to_dissolve_p, to_dissol_pn_no_neigh), axis=1)

        self.coord_buffer.reset_buffer()
        self.to_dissol_pn_buffer.reset_buffer()

        if len(to_dissolve[0]) > 0:
            counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]

            self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
            self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
            self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts

            to_dissolve = np.repeat(to_dissolve, counts, axis=1)
            self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
            new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
            new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
            new_dirs -= 1
            self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def precipitation_0(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index
        self.primary_oxidant.transform_to_3d(furthest_index, self.curr_max_furthest)
        if self.iteration % self.param["stride"] == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        oxidant_mass = oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(furthest_index + 1)], dtype=np.uint32)
        active_mass = active * self.param["active_element"]["primary"]["mass_per_cell"]

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        product_mass = product * self.param["product"]["primary"]["mass_per_cell"]

        pure_matrix = self.cells_per_page * self.param["product"]["primary"]["oxidation_number"]\
                      - active - product
        less_than_zero = np.where(pure_matrix < 0)[0]
        pure_matrix[less_than_zero] = 0
        matrix_mass = pure_matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

        whole = matrix_mass + oxidant_mass + product_mass + active_mass

        # whole_mole = matrix_mole + active_mole + oxidant_mole
        # oxidant_m_P = oxidant_mole / whole_mole
        # active_m_P = active_mole / whole_mole
        # cr = 1.10521619209865
        # m = 1 / 1.5
        # oxidant_1 = oxidant / (oxidant + m * (active + cr * (self.cells_per_page - active)))
        # n_cor_cr = self.param["matrix_elem"]["moles_per_cell"] / self.param["active_element"]["primary"]["moles_per_cell"]
        # active_1 = active / ((oxidant * 1.5) + self.cells_per_page * n_cor_cr + active * (1 - cr))
        solub_prod = (oxidant_mass * active_mass) / (whole ** 2)

        plane_indexes = np.array(np.where(solub_prod >= self.param["sol_prod"])[0])

        if len(plane_indexes) > 0:
            # self.fix_init_precip(furthest_index, self.primary_product, self.primary_oxidant.cut_shape)
            self.precip_step(plane_indexes)
        self.primary_oxidant.transform_to_descards()

    def get_combi_ind(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        # last_nonzero_index = np.max(np.nonzero(product), initial=0)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        prod_fraction = product / self.cells_per_page
        self.rel_prod_fraction = prod_fraction / self.param["phase_fraction_lim"]
        self.product_indexes = np.where(prod_fraction <= self.param["phase_fraction_lim"])[0]
        # self.product_indexes = product_indexes[:last_nonzero_index + 2]

        # if self.iteration == 0:
        #     product_indexes = np.where(prod_fraction < 0.1)[0]
        # else:
        #     product_indexes = np.where((prod_fraction < 0.1) & (prod_fraction > 0))[0]

        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

    def precipitation_0_cells(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d(self.furthest_index)

        if self.iteration % self.param["stride"] == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        # oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
        #                     in range(furthest_index + 1)], dtype=np.uint32)
        # active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
        #                    in range(furthest_index + 1)], dtype=np.uint32)
        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in range(furthest_index + 1)], dtype=np.uint32)
        #
        # last_nonzero_index = np.max(np.nonzero(product), initial=0)
        #
        # oxidant_indexes = np.where(oxidant > 0)[0]
        # active_indexes = np.where(active > 0)[0]
        # prod_fraction = product / self.cells_per_page
        # rel_prod_fraction = prod_fraction / self.param["phase_fraction_lim"]
        # product_indexes = np.where(prod_fraction <= self.param["phase_fraction_lim"])[0]
        # product_indexes = product_indexes[:last_nonzero_index + 2]
        # # if self.iteration == 0:
        # #     product_indexes = np.where(prod_fraction < 0.1)[0]
        # # else:
        # #     product_indexes = np.where((prod_fraction < 0.1) & (prod_fraction > 0))[0]
        #
        # min_act = active_indexes.min(initial=self.cells_per_axis)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act - 1)[0]
        #     comb_indexes = oxidant_indexes[indexs]
        #     comb_indexes = np.intersect1d(comb_indexes, product_indexes)
        #
        # else:
        #     comb_indexes = [furthest_index]
        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            # if self.iteration == 0:
            #     # print("product_fraction: ", rel_prod_fraction[comb_indexes])
            #     self.nucl_prob.adapt_hf_n_neigh_nucl_prob(self.comb_indexes, self.rel_prod_fraction[self.comb_indexes])
            #     self.fix_init_precip(self.furthest_index, self.primary_product)
            #     self.precip_step(self.comb_indexes)
            #     # self.probabilities.reset_constants(10**-10, 10**10, self.param["hf_deg_lim"])
            #     # print(np.sum(self.primary_product.c3d))
            #
            # else:

            self.nucl_prob.adapt_hf_n_neigh_nucl_prob(self.comb_indexes, self.rel_prod_fraction[self.comb_indexes])
            self.fix_init_precip(self.furthest_index, self.primary_product)
            self.precip_step(self.comb_indexes)
            # print(np.sum(self.primary_product.c3d))

            # self.primary_oxidant.transform_to_descards()
            # decomp_ind = np.array(np.where(prod_fraction[comb_indexes] >= 0.1)[0])
            #
            # if len(decomp_ind) > 0:
            #     self.decomposition_0(comb_indexes[decomp_ind])


            # print(np.sum(self.primary_product.c3d))

        self.primary_oxidant.transform_to_descards()

    def dissolution_0_cells(self):
        self.get_combi_ind()
        self.dissol_prob.adapt_hf_n_neigh_dissol_prob(self.product_indexes, self.rel_prod_fraction[self.product_indexes])
        self.decomposition_adapted(self.product_indexes)

    def precipitation_1(self):
        # ONE oxidant and TWO active elements exist. TWO products can be created.
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index
        self.primary_oxidant.transform_to_3d(furthest_index)

        if self.iteration % self.param["stride"] == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        primary_oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        primary_oxidant_mass = primary_oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]

        primary_active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                                   in range(furthest_index + 1)], dtype=np.uint32)
        primary_active_mass = primary_active * self.param["active_element"]["primary"]["mass_per_cell"]

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(furthest_index + 1)], dtype=np.uint32)
        secondary_active_mass = secondary_active * self.param["active_element"]["secondary"]["mass_per_cell"]

        primary_product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]

        matrix_mass = (self.param["n_cells_per_axis"] ** 2 - secondary_active - primary_active - secondary_product -
                       primary_product) * \
                      self.param["matrix_elem"]["mass_per_cell"]

        whole = matrix_mass + primary_oxidant_mass + secondary_product_mass + primary_product_mass + \
                secondary_active_mass + primary_active_mass
        primary_solub_prod = (primary_oxidant_mass * primary_active_mass) / whole ** 2
        secondary_solub_prod = (primary_oxidant_mass * secondary_active_mass) / whole ** 2

        # if primary_solub_prod >= self.param["sol_prod"]:
        #     self.case = 0
        #     self.precip_step(plane_x_ind)
        #
        # if secondary_solub_prod >= self.param["sol_prod"]:
        #     self.case = 1
        #     self.precip_step(plane_x_ind)

    def precipitation_1_cells(self):
        # ONE oxidant and TWO active elements exist. TWO products can be created.
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index

        self.primary_oxidant.transform_to_3d(furthest_index)

        if self.iteration % self.param["stride"] == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        primary_active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                                   in range(furthest_index + 1)], dtype=np.uint32)
        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(furthest_index + 1)], dtype=np.uint32)
        primary_product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                                    in range(furthest_index + 1)], dtype=np.uint32)
        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(furthest_index + 1)], dtype=np.uint32)

        presumm = oxidant + primary_product + secondary_product
        al_prim_act = primary_active + primary_product
        al_sec_act = secondary_active + secondary_product
        n_cor_o = self.param["matrix_elem"]["moles_per_cell"] / self.param["oxidant"]["primary"]["moles_per_cell"]
        n_cor_cr = self.param["matrix_elem"]["moles_per_cell"] / self.param["active_element"]["primary"]["moles_per_cell"]
        n_cor_al = self.param["matrix_elem"]["moles_per_cell"] / self.param["active_element"]["secondary"][
            "moles_per_cell"]
        oxidant_mole = presumm / (presumm + self.cells_per_page * n_cor_o +
                                  self.param["active_element"]["primary"]["n_ELEM"] * al_prim_act / 1.5 +
                                  self.param["active_element"]["secondary"]["n_ELEM"] * al_sec_act / 1.5)
        p_active_mole = al_prim_act / (1.5 * presumm + self.cells_per_page * n_cor_cr +
                                       self.param["active_element"]["primary"]["n_ELEM"] * al_prim_act +
                                       self.param["active_element"]["secondary"]["n_ELEM"] * al_sec_act)
        s_active_mole = al_sec_act / (1.5 * presumm + self.cells_per_page * n_cor_al +
                                      self.param["active_element"]["primary"]["n_ELEM"] * al_prim_act +
                                      self.param["active_element"]["secondary"]["n_ELEM"] * al_sec_act)

        oxidant_indexes = np.where(oxidant > 0)[0]
        primary_active_indexes = np.where(primary_active > 1)[0]
        primary_active_indexes_neg =\
            np.delete(range(furthest_index + 1), primary_active_indexes)
        secondary_active_indexes = np.where(secondary_active > 1)[0]

        min_act = primary_active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act)[0]
            primary_comb_indexes = oxidant_indexes[indexs]
        else:
            primary_comb_indexes = [furthest_index]

        if len(primary_comb_indexes) > 0:
            self.case = 0
            self.precip_step(primary_comb_indexes)

        if len(primary_active_indexes_neg) > 0:
            secondary_active_indexes = np.array(np.intersect1d(secondary_active_indexes, primary_active_indexes_neg))
            min_act = secondary_active_indexes.min(initial=self.cells_per_axis + 10)
            if min_act < self.cells_per_axis:
                indexs = np.where(oxidant_indexes >= min_act)[0]
                secondary_comb_indexes = oxidant_indexes[indexs]
            else:
                secondary_comb_indexes = [furthest_index]

            if len(secondary_comb_indexes) > 0:
                self.case = 1
                self.precip_step(secondary_comb_indexes)

        self.primary_oxidant.transform_to_descards()

    def precipitation_2_cells(self):
        # TWO oxidant and TWO active elements exist. FOUR products can be created.
        primary_furthest_index = self.primary_oxidant.calc_furthest_index()
        secondary_furthest_index = self.secondary_oxidant.calc_furthest_index()

        if primary_furthest_index > secondary_furthest_index:
            furthest_index = primary_furthest_index
        else:
            furthest_index = secondary_furthest_index

        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index

        self.primary_oxidant.transform_to_3d(furthest_index)
        self.secondary_oxidant.transform_to_3d(furthest_index)

        if self.iteration % self.param["stride"] == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)

        secondary_oxidant = np.array([np.sum(self.secondary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                                      in range(furthest_index + 1)], dtype=np.uint32)
        primary_active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                                   in range(furthest_index + 1)], dtype=np.uint32)
        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(furthest_index + 1)], dtype=np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        secondary_oxidant_indexes = np.where(secondary_oxidant > 0)[0]
        primary_active_indexes = np.where(primary_active > 1)[0]
        secondary_active_indexes = np.where(secondary_active > 1)[0]

        min_act = primary_active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act)[0]
            primary_comb_indexes = oxidant_indexes[indexs]
        else:
            primary_comb_indexes = [furthest_index]

        min_act = secondary_active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act)[0]
            secondary_comb_indexes = oxidant_indexes[indexs]
        else:
            secondary_comb_indexes = [furthest_index]

        min_act = primary_active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(secondary_oxidant_indexes >= min_act)[0]
            ternary_comb_indexes = secondary_oxidant_indexes[indexs]
        else:
            ternary_comb_indexes = [furthest_index]

        min_act = secondary_active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(secondary_oxidant_indexes >= min_act)[0]
            quaternary_comb_indexes = secondary_oxidant_indexes[indexs]
        else:
            quaternary_comb_indexes = [furthest_index]

        if len(primary_comb_indexes) > 0:
            self.case = 0
            self.precip_step(primary_comb_indexes)

        if len(secondary_comb_indexes) > 0:
            self.case = 1
            self.precip_step(secondary_comb_indexes)

        if len(ternary_comb_indexes) > 0:
            self.case = 2
            self.precip_step(ternary_comb_indexes)

        if len(quaternary_comb_indexes) > 0:
            self.case = 3
            self.precip_step(quaternary_comb_indexes)

        self.primary_oxidant.transform_to_descards()
        self.secondary_oxidant.transform_to_descards()

    def precip_step(self, indexes):
        for plane_index in reversed(indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.objs[self.case]["oxidant"].c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.objs[self.case]["product"].full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        # activate if microstructure ___________________________________________________________
                        # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                        # temp_ind = np.where(in_gb)[0]
                        # oxidant_cells = oxidant_cells[temp_ind]
                        # ______________________________________________________________________________________
                        self.check_intersection(oxidant_cells)

    def ci_single(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.objs[self.case]["active"].c3d.shape[2] - 1)
        neighbours = go_around_bool(self.objs[self.case]["active"].c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            flat_arounds = all_arounds[:, 0:self.objs[self.case]["product"].lind_flat_arr]

            # flat_neighbours = self.go_around(self.precipitations3d_init_full, flat_arounds)
            # flat_neighbours = self.go_around(flat_arounds)
            # arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)

            arr_len_in_flat = self.go_around(self.precipitations3d_init, flat_arounds)

            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]

            needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])

            # if self.iteration < 10000:
            #     needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])
            # else:
            #     needed_prob = self.nucl_prob.get_probabilities_from_map(arr_len_in_flat)

            needed_prob[homogeneous_ind] = self.nucl_prob.nucl_prob_pp[seeds[0][2]]  # seeds[0][2] - current plane index
            randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
            temp_ind = np.where(randomise < needed_prob)[0]
        # _________________________________________________________________________________________________

            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]
                out_to_del = np.array(np.nonzero(neighbours))
                start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                                  dtype=np.ubyte)
                coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)
                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))

                # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, coord)  # precip on place of active!
                # exists = check_at_coord(self.objs[self.case]["product"].full_c3d, seeds)  # precip on place of oxidant!

                # temp_ind = np.where(exists)[0]
                # coord = np.delete(coord, temp_ind, 0)
                # seeds = np.delete(seeds, temp_ind, 0)

                # if self.objs[self.case]["to_check_with"] is not None:
                #     # to_check_min_self = np.array(self.cumul_product - product.c3d, dtype=np.ubyte)
                #     exists = np.array([self.objs[self.case]["to_check_with"].c3d[point[0], point[1], point[2]]
                #                        for point in coord], dtype=np.ubyte)
                #     # exists = np.array([to_check_min_self[point[0], point[1], point[2]] for point in coord],
                #     #                   dtype=np.ubyte)
                #     temp_ind = np.where(exists > 0)[0]
                #     coord = np.delete(coord, temp_ind, 0)
                #     seeds = np.delete(seeds, temp_ind, 0)

                coord = coord.transpose()
                seeds = seeds.transpose()

                self.objs[self.case]["active"].c3d[coord[0], coord[1], coord[2]] -= 1
                self.objs[self.case]["oxidant"].c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.objs[self.case]["product"].c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.objs[self.case]["product"].fix_full_cells(seeds)  # precip on place of oxidant!

                # self.cumul_product[coord[0], coord[1], coord[2]] += 1

    def ci_multi(self, seeds):
        """
        Check intersections between the seeds neighbourhood and the coordinates of inward particles only.
        Compute which seed will become a precipitation and which inward particles should be deleted
        according to threshold_inward conditions. This is a simplified version of the check_intersection() function
        where threshold_outward is equal to 1, so there is no need to check intersection with OUT arrays!

        :param seeds: array of seeds coordinates
        """
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.objs[self.case]["active"].c3d.shape[2] - 1)
        neighbours = np.array([[self.objs[self.case]["active"].c3d[point[0], point[1], point[2]]
                                for point in seed_arrounds] for seed_arrounds in all_arounds], dtype=bool)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            out_to_del = np.array(np.nonzero(neighbours))
            start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
            to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                              dtype=int)
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)
            # coord = np.reshape(coord, (len(coord) * self.threshold_inward, 3)).transpose()
            coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
            exists = [self.objs[self.case]["product"].full_c3d[point[0], point[1], point[2]] for point in coord]
            temp_ind = np.where(exists)[0]
            coord = np.delete(coord, temp_ind, 0)
            seeds = np.delete(seeds, temp_ind, 0)

            if self.objs[self.case]["to_check_with"] is not None:
                exists = [self.objs[self.case]["to_check_with"].c3d[point[0], point[1], point[2]] for point in coord]
                temp_ind = np.where(exists)[0]
                coord = np.delete(coord, temp_ind, 0)
                seeds = np.delete(seeds, temp_ind, 0)

            if len(seeds) > 0:
                self_all_arounds = self.utils.calc_sur_ind_formation_noz(seeds, self.objs[self.case]["oxidant"].c3d.shape[2] - 1)
                self_neighbours = np.array([[self.objs[self.case]["oxidant"].c3d[point[0], point[1], point[2]]
                                             for point in seed_arrounds]
                                            for seed_arrounds in self_all_arounds], dtype=bool)
                arr_len_in = np.array([np.sum(item) for item in self_neighbours], dtype=np.ubyte)
                temp_ind = np.where(arr_len_in >= self.threshold_inward)[0]
                # if len(index_in) > 0:
                #     seeds = seeds[index_in]
                #     neighbours = neighbours[index_in]
                #     all_arounds = all_arounds[index_in]
                #     flat_arounds = all_arounds[:, 0:6]
                #     flat_neighbours = np.array(
                #         [[self.precipitations3d_init[point[0], point[1], point[2]] for point in seed_arrounds]
                #          for seed_arrounds in flat_arounds], dtype=bool)
                #     arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)
                #     homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
                #     needed_prob = self.const_a * 2.718281828 ** (self.const_b * arr_len_in_flat)
                #     needed_prob[homogeneous_ind] = self.scale_probability
                #     randomise = np.random.random_sample(len(arr_len_in_flat))
                #     temp_ind = np.where(randomise < needed_prob)[0]

                if len(temp_ind) > 0:
                    seeds = seeds[temp_ind]
                    coord = coord[temp_ind]

                    # neighbours = neighbours[temp_ind]
                    # all_arounds = all_arounds[temp_ind]

                    self_neighbours = self_neighbours[temp_ind]
                    self_all_arounds = self_all_arounds[temp_ind]

                    in_to_del = np.array(np.nonzero(self_neighbours))
                    in_start_seed_index = np.unique(in_to_del[0], return_index=True)[1]
                    to_del_in = np.array(
                        [in_to_del[1, indx:indx + self.threshold_inward - 1] for indx in in_start_seed_index],
                        dtype=int)
                    coord_in = np.array([self_all_arounds[seed_ind][point_ind] for seed_ind, point_ind in
                                         enumerate(to_del_in)], dtype=np.short)
                    coord_in = np.reshape(coord_in, (len(coord_in) * (self.threshold_inward - 1), 3)).transpose()

                    # out_to_del = np.array(np.nonzero(neighbours))
                    # start_seed_index = np.unique(out_to_del[0], return_index=True)[1]
                    # to_del = np.array([out_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                    #                   dtype=int)
                    # coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                    #                  dtype=np.short)
                    # # coord = np.reshape(coord, (len(coord) * self.threshold_inward, 3)).transpose()
                    # coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                    # exists = [product.full_c3d[point[0], point[1], point[2]] for point in coord]
                    # temp_ind = np.where(exists)[0]
                    # coord = np.delete(coord, temp_ind, 0)
                    # seeds = np.delete(seeds, temp_ind, 0)
                    #
                    # if to_check_with is not None:
                    #     exists = [to_check_with.c3d[point[0], point[1], point[2]] for point in coord]
                    #     temp_ind = np.where(exists)[0]
                    #     coord = np.delete(coord, temp_ind, 0)
                    #     seeds = np.delete(seeds, temp_ind, 0)

                    coord = coord.transpose()
                    seeds = seeds.transpose()

                    self.objs[self.case]["active"].c3d[coord[0], coord[1], coord[2]] -= 1
                    self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1
                    self.objs[self.case]["product"].fix_full_cells(coord)

                    self.objs[self.case]["oxidant"].c3d[seeds[0], seeds[1], seeds[2]] -= 1
                    self.objs[self.case]["oxidant"].c3d[coord_in[0], coord_in[1], coord_in[2]] -= 1

    def diffusion_inward(self):
        self.primary_oxidant.diffuse()
        if self.param["secondary_oxidant_exists"]:
            self.secondary_oxidant.diffuse()

    def diffusion_outward(self):
        if (self.iteration + 1) % self.param["stride"] == 0:
            self.primary_active.transform_to_descards()
            self.primary_active.diffuse()
            if self.param["secondary_active_element_exists"]:
                self.secondary_active.transform_to_descards()
                self.secondary_active.diffuse()

    def calc_precip_front_0(self, iteration):
        """
        Calculating a position of a precipitation front. As a boundary a precipitation concentration of 0,1% is used.
        :param iteration: current iteration (serves as a current simulation time)
        """
        oxidant = self.primary_oxidant.cells
        oxidant = np.array([len(np.where(oxidant[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        oxidant_mass = oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]
        active = self.primary_active.cells
        active = np.array([len(np.where(active[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        active_mass = active * self.param["active_element"]["primary"]["mass_per_cell"]

        # primary_product = np.array(np.nonzero(self.primary_product.c3d), dtype=int)
        # primary_product = np.array([len(np.where(primary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
        #                            dtype=int)

        primary_product = np.array([np.sum(self.primary_product.c3d[:, :, i]) for i in range(self.cells_per_axis)],
                                   dtype=int)
        precipitations_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]

        pure_matrix = (self.param["n_cells_per_axis"] ** 2) * self.param["product"]["primary"]["oxidation_number"] \
                      - active - primary_product
        less_than_zero = np.where(pure_matrix < 0)[0]
        pure_matrix[less_than_zero] = 0
        matrix_mass = pure_matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

        whole_mass = matrix_mass + oxidant_mass + precipitations_mass + active_mass
        concentration = precipitations_mass / whole_mass

        for rev_index, precip_conc in enumerate(np.flip(concentration)):
            if precip_conc > 0.001:
                position = (self.cells_per_axis - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def calc_precip_front_1(self, iteration):
        """
        Calculating a position of a precipitation front. As a boundary a precipitation concentration of 0,1% is used.
        :param iteration: current iteration (serves as a current simulation time)
        """
        oxidant = self.primary_oxidant.cells
        oxidant = np.array([len(np.where(oxidant[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        oxidant_mass = oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]

        active = self.primary_active.cells
        active = np.array([len(np.where(active[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        active_mass = active * self.param["active_element"]["primary"]["mass_per_cell"]

        secondary_active = self.secondary_active.cells
        secondary_active = np.array([len(np.where(secondary_active[2] == i)[0]) for i in range(self.cells_per_axis)],
                                    dtype=int)
        secondary_active_mass = secondary_active * self.param["active_element"]["secondary"]["mass_per_cell"]

        primary_product = np.array(np.nonzero(self.primary_product.c3d), dtype=int)
        primary_product = np.array([len(np.where(primary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                   dtype=int)
        primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]

        secondary_product = np.array(np.nonzero(self.secondary_product.c3d), dtype=int)
        secondary_product = np.array([len(np.where(secondary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                     dtype=int)
        secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]

        matrix_mass = (self.param[
                           "n_cells_per_axis"] ** 2 - active - secondary_active - primary_product - secondary_product) * \
                      self.param["matrix_elem"]["mass_per_cell"]
        whole_mass = matrix_mass + oxidant_mass + primary_product_mass + active_mass + secondary_active_mass + \
                     secondary_product_mass
        primary_product_concentration = primary_product_mass / whole_mass
        secondary_product_concentration = secondary_product_mass / whole_mass

        for rev_index, precip_conc in enumerate(np.flip(primary_product_concentration)):
            if precip_conc > 0.001:
                position = (self.cells_per_axis - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

        for rev_index, precip_conc in enumerate(np.flip(secondary_product_concentration)):
            if precip_conc > 0.001:
                position = (self.cells_per_axis - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "s")
                break

    def calc_precip_front_2(self, iteration):
        """
        Calculating a position of a precipitation front. As a boundary a precipitation concentration of 0,1% is used.
        :param iteration: current iteration (serves as a current simulation time)
        """
        oxidant = self.primary_oxidant.cells
        oxidant = np.array([len(np.where(oxidant[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        oxidant_mass = oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]

        active = self.primary_active.cells
        active = np.array([len(np.where(active[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        active_mass = active * self.param["active_element"]["primary"]["mass_per_cell"]

        secondary_active = self.secondary_active.cells
        secondary_active = np.array([len(np.where(secondary_active[2] == i)[0]) for i in range(self.cells_per_axis)],
                                    dtype=int)
        secondary_active_mass = secondary_active * self.param["active_element"]["secondary"]["mass_per_cell"]

        primary_product = np.array(np.nonzero(self.primary_product.c3d), dtype=int)
        primary_product = np.array([len(np.where(primary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                   dtype=int)
        primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]

        secondary_product = np.array(np.nonzero(self.secondary_product.c3d), dtype=int)
        secondary_product = np.array([len(np.where(secondary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                     dtype=int)
        secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]

        matrix_mass = (self.param[
                           "n_cells_per_axis"] ** 2 - active - secondary_active - primary_product - secondary_product) * \
                      self.param["matrix_elem"]["mass_per_cell"]
        whole_mass = matrix_mass + oxidant_mass + primary_product_mass + active_mass + secondary_active_mass + \
                     secondary_product_mass
        primary_product_concentration = primary_product_mass / whole_mass
        secondary_product_concentration = secondary_product_mass / whole_mass

        for rev_index, precip_conc in enumerate(np.flip(primary_product_concentration)):
            if precip_conc > 0.001:
                position = (self.cells_per_axis - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

        for rev_index, precip_conc in enumerate(np.flip(secondary_product_concentration)):
            if precip_conc > 0.001:
                position = (self.cells_per_axis - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "s")
                break

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 0,1% is used.

        :param iteration: current iteration (serves as a current simulation time).
        """

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.cells_per_axis)], dtype=np.uint32)

        product = product / (self.param["n_cells_per_axis"] ** 2)
        threshold = self.param["active_element"]["primary"]["cells_concentration"]

        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 100:
                position = (len(product) - 1 - rev_index) * self.utils.param["size"] * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((self.iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def save_results(self):
        if self.param["stride"] > self.param["n_iterations"]:
            self.primary_active.transform_to_descards()
            if self.param["secondary_active_element_exists"]:
                self.secondary_active.transform_to_descards()

        if self.param["inward_diffusion"]:
            self.utils.db.insert_particle_data("primary_oxidant", self.iteration, self.primary_oxidant.cells)
            if self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_oxidant", self.iteration, self.secondary_oxidant.cells)

        if self.param["outward_diffusion"]:
            self.utils.db.insert_particle_data("primary_active", self.iteration, self.primary_active.cells)
            if self.param["secondary_active_element_exists"]:
                self.utils.db.insert_particle_data("secondary_active", self.iteration, self.secondary_active.cells)

        if self.param["compute_precipitations"]:
            self.utils.db.insert_particle_data("primary_product", self.iteration, self.primary_product.transform_c3d())

            if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_product", self.iteration,
                                                   self.secondary_product.transform_c3d())
                self.utils.db.insert_particle_data("ternary_product", self.iteration,
                                                   self.ternary_product.transform_c3d())
                self.utils.db.insert_particle_data("quaternary_product", self.iteration,
                                                   self.quaternary_product.transform_c3d())

            elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_product", self.iteration, self.secondary_product.transform_c3d())

        if self.param["stride"] > self.param["n_iterations"]:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            if self.param["secondary_active_element_exists"]:
                self.secondary_active.transform_to_3d(self.curr_max_furthest)

    def save_results_only_prod(self):
        self.utils.db.insert_particle_data("primary_product", self.iteration,
                                           self.primary_product.transform_c3d())

    def save_results_prod_and_inw(self):
        self.utils.db.insert_particle_data("primary_product", self.iteration,
                                           self.primary_product.transform_c3d())
        self.utils.db.insert_particle_data("primary_oxidant", self.iteration, self.primary_oxidant.cells)

    def save_results_only_inw(self):
        self.utils.db.insert_particle_data("primary_oxidant", self.iteration, self.primary_oxidant.cells)

    def fix_init_precip_bool(self, u_bound, product):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        self.precipitations3d_init[:, :, 0:u_bound + 2] = False
        self.precipitations3d_init[:, :, 0:u_bound + 2] = product.c3d[:, :, 0:u_bound + 2]

    def fix_init_precip_int(self, u_bound, product):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        self.precipitations3d_init[:, :, 0:u_bound + 2] = 0
        self.precipitations3d_init[:, :, 0:u_bound + 2] = product.c3d[:, :, 0:u_bound + 2]

    def go_around_single_oxid_n(self, around_coords):
        flat_neighbours = go_around_bool(self.precipitations3d_init, around_coords)
        return np.array([np.sum(item) for item in flat_neighbours], dtype=int)

    def go_around_mult_oxid_n(self, array_3d, around_coords):
        all_neigh = go_around_int(array_3d, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()

        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        where_full = np.unique(np.where(all_neigh[:, :6].view() == self.primary_oxid_numb)[0])

        only_inside_product_program = np.setdiff1d(nonzero_neigh_in_prod, where_full, assume_unique=True)

        fanal_effective_flat_counts = np.zeros(len(all_neigh), dtype=np.single)
        fanal_effective_flat_counts[where_full] = np.sum(all_neigh[where_full], axis=1)
        fanal_effective_flat_counts[only_inside_product_program] =\
            neigh_in_prod[only_inside_product_program] * (7 * self.primary_oxid_numb - 1) / (self.primary_oxid_numb - 1)

        return fanal_effective_flat_counts

    def generate_fetch_ind(self):
        size = 3 + (self.param["neigh_range"] - 1) * 2
        if self.cells_per_axis % size == 0:
            length = int((self.cells_per_axis / size) ** 2)
            self.fetch_ind = np.zeros((size**2, 2, length), dtype=np.short)
            iter_shifts = np.array(np.where(np.ones((size, size)) == 1)).transpose()
            dummy_grid = np.full((self.cells_per_axis, self.cells_per_axis), True)
            all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
            for step, t in enumerate(iter_shifts):
                t_ind = np.where(((all_coord[0] - t[1]) % size == 0) & ((all_coord[1] - t[0]) % size == 0))[0]
                self.fetch_ind[step] = all_coord[:, t_ind]
        else:
            print()
            print("______________________________________________________________")
            print("Number of Cells per Axis must be divisible by ", size, "!!!")
            print("______________________________________________________________")
            sys.exit()

        # length = int((self.cells_per_axis / 3) ** 2)
        # self.fetch_ind = np.zeros((9, 2, length), dtype=np.short)
        # iter_shifts = np.array(np.where(np.ones((3, 3)) == 1)).transpose()
        # dummy_grid = np.full((self.cells_per_axis, self.cells_per_axis), True)
        # all_coord = np.array(np.nonzero(dummy_grid), dtype=np.short)
        # for step, t in enumerate(iter_shifts):
        #     t_ind = np.where(((all_coord[0] - t[1]) % 3 == 0) & ((all_coord[1] - t[0]) % 3 == 0))[0]
        #     self.fetch_ind[step] = all_coord[:, t_ind]

    def insert_last_it(self):
        self.utils.db.insert_last_iteration(self.iteration)
