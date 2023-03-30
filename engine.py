import numpy as np

from utils.utilities import *
import gc
import progressbar
import elements
import time


class CellularAutomata:
    """
    TODO: 1 -> Write comments.
          2 -> Check aggregated indexes!
    """

    def __init__(self, user_input=None):
        # pre settings
        if user_input is None:
            # setting default parameters if no user input is given
            user_input = physical_data.DEFAULT_PARAM
        self.utils = Utils(user_input)
        self.utils.create_database()
        self.utils.generate_param()
        self.utils.print_init_var()
        self.param = self.utils.param
        self.iteration = None
        self.curr_max_furthest = 0

        self.begin = time.time()
        self.elapsed_time = 0
        # simulated space parameters
        self.cells_per_axis = self.param["n_cells_per_axis"]
        self.n_iter = self.param["n_iterations"]
        self.shape = (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis)
        self.cut_shape = (self.cells_per_axis, self.cells_per_axis, 3)
        self.single_page_shape = (self.cells_per_axis, self.cells_per_axis)

        self.obj_ref = physical_data.DEFAULT_OBJ_REF

        # setting variables for inward diffusion
        if self.param["inward_diffusion"]:
            self.primary_oxidant = elements.OxidantElem(self.param["oxidant"]["primary"])
            self.obj_ref[0]["oxidant"] = self.primary_oxidant
            self.obj_ref[1]["oxidant"] = self.primary_oxidant
            if self.param["secondary_oxidant_exists"]:
                self.secondary_oxidant = elements.OxidantElem(self.param["oxidant"]["secondary"])
                self.obj_ref[2]["oxidant"] = self.primary_oxidant
                self.obj_ref[3]["oxidant"] = self.primary_oxidant
        # setting variables for outward diffusion
        if self.param["outward_diffusion"]:
            self.primary_active = elements.ActiveElem(self.param["active_element"]["primary"])
            self.obj_ref[0]["active"] = self.primary_active
            self.obj_ref[2]["active"] = self.primary_active
            if self.param["secondary_active_element_exists"]:
                self.secondary_active = elements.ActiveElem(self.param["active_element"]["secondary"])
                self.obj_ref[1]["active"] = self.secondary_active
                self.obj_ref[3]["active"] = self.secondary_active
        # setting variables for precipitations
        if self.param["compute_precipitations"]:
            self.primary_product = elements.Product(self.param["product"]["primary"])
            self.obj_ref[0]["product"] = self.primary_product
            if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                self.secondary_product = elements.Product(self.param["product"]["secondary"])
                self.obj_ref[1]["product"] = self.secondary_product
                self.ternary_product = elements.Product(self.param["product"]["ternary"])
                self.obj_ref[2]["product"] = self.ternary_product
                self.quaternary_product = elements.Product(self.param["product"]["quaternary"])
                self.obj_ref[3]["product"] = self.quaternary_product

                self.precip_func = self.precipitation_2_cells
                self.calc_precip_front = self.calc_precip_front_2
                self.decomposition = self.decomposition_2

            elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                self.secondary_product = elements.Product(self.param["product"]["secondary"])
                self.obj_ref[1]["product"] = self.secondary_product
                self.precip_func = self.precipitation_1_cells
                self.calc_precip_front = self.calc_precip_front_1
                self.decomposition = self.decomposition_1

            else:
                self.precip_func = self.precipitation_0_cells
                # self.precip_func = self.precipitation_0
                self.calc_precip_front = self.calc_precip_front_0
                self.decomposition = self.decomposition_0

            self.precipitations = None
            self.cumul_product = np.full(self.shape, 0, dtype=np.ubyte)
            self.precipitations3d_init = None

            # self.half_thickness = 20
            # middle = int(self.cells_per_axis / 2)
            # minus = middle - self.half_thickness
            # plus = middle + self.half_thickness
            # self.primary_product.c3d = np.zeros(self.shape, dtype=int)
            # self.primary_product.c3d[minus:plus, minus:plus, minus:plus] = 1
            # shift = 20
            # self.precipitations[minus + shift:plus + shift, minus + shift:plus + shift, minus + shift:plus + shift] = 1
            # self.precipitations = np.array(np.nonzero(self.precipitations), dtype=int)
            # self.precipitations3d = np.full(self.single_page_shape, False)
            # self.precipitations3d_sec = np.full(self.single_page_shape, False)

            self.threshold_inward = self.param["threshold_inward"]
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
            self.nucleation_probability = self.param["nucleation_probability"]
            self.het_factor = self.param["het_factor"]
            self.const_a = (1 / (self.het_factor * self.nucleation_probability)) ** (-6 / 5)
            self.const_b = log(1 / (self.het_factor * self.nucleation_probability)) * (1 / 5)

    def simulation(self):
        """
        Simulation of diffusion and precipitation formation.
        """
        # self.primary_oxidant.transform_to_3d(0)
        # self.primary_active.transform_to_3d(0)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # self.precip_step([0], self.primary_oxidant, self.primary_active, self.primary_product)
        # return np.sum(self.primary_product.c3d)
        for self.iteration in progressbar.progressbar(range(self.n_iter)):
            if self.param["compute_precipitations"]:
                self.precip_func()
                # self.decomposition_0()
            if self.param["inward_diffusion"]:
                self.diffusion_inward()
            if self.param["outward_diffusion"]:
                self.diffusion_outward()
            self.utils.db.insert_last_iteration(self.iteration)
            if self.param["save_whole"] and self.iteration != self.n_iter - 1:
                self.save_results(self.iteration)

        self.save_results(self.n_iter - 1)
        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.utils.db.insert_time(self.elapsed_time)
        self.utils.db.conn.commit()

    def decomposition_0(self):
        self.precipitations = self.primary_product.transform_c3d()
        if len(self.precipitations[0] > 0):
            dec_p_three_open = np.array([[], [], []], dtype=np.short)
            dec_p_two_open = np.array([[], [], []], dtype=np.short)
            dec_p_one_open = np.array([[], [], []], dtype=np.short)
            dec_pn = np.array([[], [], []], dtype=np.short)
            arr_len_flat_to_pn = np.array([[], [], []], dtype=np.ubyte)

            all_arounds = self.utils.calc_sur_ind_decompose(self.precipitations)
            neighbours = np.array([[self.primary_product.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
                                   for seed_arrounds in all_arounds], dtype=bool)

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
                # else:
                #     dec_p_three_open = np.array([[], [], []], dtype=int)
                #     dec_p_two_open = np.array([[], [], []], dtype=int)
                #     dec_p_one_open = np.array([[], [], []], dtype=int)

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

            if len(todec[0] > 0):
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, todec), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(todec[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

                counts = np.unique(np.ravel_multi_index(todec, self.shape), return_counts=True)
                dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short)
                counts = np.array(counts[1], dtype=np.ubyte)

                self.primary_product.c3d[dec[0], dec[1], dec[2]] -= counts
                self.primary_product.full_c3d[dec[0], dec[1], dec[2]] = False
                dec[2] += 1
                self.primary_active.c3d[dec[0], dec[1], dec[2]] += counts
                # if self.iteration == 0:
                #     self.primary_active.transform_to_descards()

    def decomposition_1(self):
        self.precipitations = self.secondary_product.transform_c3d()
        if len(self.precipitations[0] > 0):
            all_arounds = self.utils.calc_sur_ind_decompose(self.precipitations)
            neighbours = np.array([[self.secondary_product.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
                                   for seed_arrounds in all_arounds], dtype=bool)
            del all_arounds
            gc.collect()
            arr_len_flat = np.array([np.sum(item[:6]) for item in neighbours], dtype=int)
            arr_len_corners = np.array([np.sum(item[6:14]) for item in neighbours], dtype=int)
            arr_len_side_corners = np.array([np.sum(item[14:]) for item in neighbours], dtype=int)
            arr_len_all = arr_len_flat + arr_len_corners + arr_len_side_corners
            # These are inside!
            # ------------------------------------------------
            index = np.where(arr_len_flat < 6)[0]
            # tonothing = np.delete(self.precipitations, index, 1)
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
            del arr_len_all
            gc.collect()
            dec_pn = np.delete(self.precipitations, index, 1)
            self.precipitations = self.precipitations[:, index]
            arr_len_flat_to_pn = np.delete(arr_len_flat, index)
            arr_len_flat = arr_len_flat[index]
            arr_len_corners = arr_len_corners[index]
            arr_len_side_corners = arr_len_side_corners[index]
            neighbours = neighbours[index]
            # ------------------------------------------------
            # These are not in a block => pn!
            # ------------------------------------------------
            index = np.where((arr_len_flat > 2) & (arr_len_corners > 0) & (arr_len_side_corners > 2))[0]
            del arr_len_corners, arr_len_side_corners
            gc.collect()
            dec_pn = np.concatenate((dec_pn, np.delete(self.precipitations, index, 1)), axis=1)
            arr_len_flat_to_pn = np.concatenate((arr_len_flat_to_pn, np.delete(arr_len_flat, index)))
            self.precipitations = self.precipitations[:, index]
            arr_len_flat = arr_len_flat[index]
            neighbours = neighbours[index]
            # ------------------------------------------------

            if np.any(self.precipitations):
                aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in neighbours],
                                       dtype=int)
                del neighbours
                gc.collect()
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
                del arr_len_flat
                gc.collect()
                dec_p_three_open = self.precipitations[:, three_open]
                dec_p_two_open = self.precipitations[:, two_open]
                dec_p_one_open = self.precipitations[:, one_open]
                # ------------------------------------------------
            else:
                dec_p_three_open = np.array([[], [], []], dtype=int)
                dec_p_two_open = np.array([[], [], []], dtype=int)
                dec_p_one_open = np.array([[], [], []], dtype=int)

            self.precipitations = np.array([[], [], []], dtype=int)
            needed_prob = self.dissol_pn * 2.718281828 ** (-self.param["exponent_power"] * arr_len_flat_to_pn)
            randomise = np.random.random_sample(len(arr_len_flat_to_pn))
            temp_ind = np.where(randomise < needed_prob)[0]
            # to_return = np.delete(dec_pn, temp_ind, 1)
            dec_pn = dec_pn[:, temp_ind]
            # self.precipitations = np.concatenate((tonothing, to_return), axis=1)

            randomise = np.random.random_sample(len(dec_p_three_open[0]))
            p = self.dissol_p_block
            temp_ind = np.where(randomise < p)[0]
            dec_p_three_open = dec_p_three_open[:, temp_ind]

            p23 = self.dissol_p_block / 10
            randomise = np.random.random_sample(len(dec_p_two_open[0]))
            temp_ind = np.where(randomise < p23)[0]
            dec_p_two_open = dec_p_two_open[:, temp_ind]

            p13 = self.dissol_p_block / 100
            randomise = np.random.random_sample(len(dec_p_one_open[0]))
            temp_ind = np.where(randomise < p13)[0]
            dec_p_one_open = dec_p_one_open[:, temp_ind]

            todec = np.concatenate((dec_pn, dec_p_three_open, dec_p_two_open, dec_p_one_open), axis=1)
            self.secondary_product.c3d[todec[0], todec[1], todec[2]] = False

            randomise = np.random.random_sample(len(todec[0]))
            temp_part = todec[:, np.where(randomise <= 0.16666666)[0]]
            self.primary_oxidant.c1 = np.concatenate((self.primary_oxidant.c1, temp_part), axis=1)
            self.secondary_active.c1 = np.concatenate((self.secondary_active.c1, temp_part), axis=1)

            temp_part = todec[:, np.where((randomise > 0.16666666) & (randomise <= 0.33333333))[0]]
            self.primary_oxidant.c2 = np.concatenate((self.primary_oxidant.c2, temp_part), axis=1)
            self.secondary_active.c2 = np.concatenate((self.secondary_active.c2, temp_part), axis=1)

            temp_part = todec[:, np.where((randomise > 0.33333333) & (randomise <= 0.5))[0]]
            self.primary_oxidant.c3 = np.concatenate((self.primary_oxidant.c3, temp_part), axis=1)
            self.secondary_active.c3 = np.concatenate((self.secondary_active.c3, temp_part), axis=1)

            temp_part = todec[:, np.where((randomise > 0.5) & (randomise <= 0.6666666))[0]]
            self.primary_oxidant.c4 = np.concatenate((self.primary_oxidant.c4, temp_part), axis=1)
            self.secondary_active.c4 = np.concatenate((self.secondary_active.c4, temp_part), axis=1)

            temp_part = todec[:, np.where((randomise > 0.6666666) & (randomise <= 0.8333333333))[0]]
            self.primary_oxidant.c5 = np.concatenate((self.primary_oxidant.c5, temp_part), axis=1)
            self.secondary_active.c5 = np.concatenate((self.secondary_active.c5, temp_part), axis=1)

            temp_part = todec[:, np.where((randomise > 0.8333333333) & (randomise < 1))[0]]
            self.primary_oxidant.c6 = np.concatenate((self.primary_oxidant.c6, temp_part), axis=1)
            self.secondary_active.c6 = np.concatenate((self.secondary_active.c6, temp_part), axis=1)

    def decomposition_2(self):
        self.decomposition_0()
        self.decomposition_1()

    def precipitation_0(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index
        self.primary_oxidant.transform_to_3d(furthest_index)
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

        pure_matrix = (self.param["n_cells_per_axis"] ** 2) * self.param["product"]["primary"]["oxidation_number"]\
                      - active - product
        less_than_zero = np.where(pure_matrix < 0)[0]
        pure_matrix[less_than_zero] = 0
        matrix_mass = pure_matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

        whole = matrix_mass + oxidant_mass + product_mass + active_mass
        solub_prod = (oxidant_mass * active_mass) / (whole ** 2)

        plane_indexes = np.array(np.where(solub_prod >= self.param["sol_prod"])[0])

        if len(plane_indexes) > 0:
            # self.fix_init_precip(furthest_index, self.primary_product, self.primary_oxidant.cut_shape)
            self.precip_step(plane_indexes, self.primary_oxidant, self.primary_active,
                             self.primary_product)
        self.primary_oxidant.transform_to_descards()

    def precipitation_0_cells(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index

        self.primary_oxidant.transform_to_3d(furthest_index)

        if self.iteration % self.param["stride"] == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        # if self.iteration == 0:
        #     self.primary_active.transform_to_3d(self.curr_max_furthest)

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(furthest_index + 1)], dtype=np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 1)[0]

        min_act = active_indexes.min(initial=self.cells_per_axis + 10)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act)[0]
            comb_indexes = oxidant_indexes[indexs]
        else:
            comb_indexes = [furthest_index]

        if len(comb_indexes) > 0:
            # self.fix_init_precip(furthest_index, self.primary_product, self.primary_active.cut_shape)
            self.precip_step(comb_indexes, self.primary_oxidant, self.primary_active, self.primary_product)
            # new_prod = len(np.nonzero(self.primary_product.c3d)[0])
        self.primary_oxidant.transform_to_descards()

    def precipitation_1(self):
        # ONE oxidant and TWO active elements exist. TWO products can be created.
        furthest_index = self.primary_oxidant.calc_furthest_index()
        for plane_x_ind in reversed(range(0, furthest_index + 1)):
            primary_oxidant = self.primary_oxidant.count_cells_at_index(plane_x_ind)
            primary_oxidant_mass = primary_oxidant * self.param["oxidant"]["primary"]["mass_per_cell"]

            primary_active = self.primary_active.count_cells_at_index(plane_x_ind)
            primary_active_mass = primary_active * self.param["active_element"]["primary"]["mass_per_cell"]

            secondary_active = self.secondary_active.count_cells_at_index(plane_x_ind)
            secondary_active_mass = secondary_active * self.param["active_element"]["secondary"]["mass_per_cell"]

            primary_product = np.sum(self.primary_product.c3d[:, :, plane_x_ind])
            primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]

            secondary_product = np.sum(self.secondary_product.c3d[:, :, plane_x_ind])
            secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]

            matrix_mass = (self.param["n_cells_per_axis"] ** 2 - secondary_active - primary_active - secondary_product -
                           primary_product) * \
                          self.param["matrix_elem"]["mass_per_cell"]

            whole = matrix_mass + primary_oxidant_mass + secondary_product_mass + primary_product_mass + \
                    secondary_active_mass + primary_active_mass
            primary_solub_prod = (primary_oxidant_mass * primary_active_mass) / whole ** 2
            secondary_solub_prod = (primary_oxidant_mass * secondary_active_mass) / whole ** 2

            if primary_solub_prod >= self.param["sol_prod"]:
                self.precip_step(plane_x_ind, self.primary_oxidant, self.primary_active, self.primary_product,
                                 self.secondary_product)

            if secondary_solub_prod >= self.param["sol_prod"]:
                self.precip_step(plane_x_ind, self.primary_oxidant, self.secondary_active, self.secondary_product,
                                 self.primary_product)

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

        oxidant_indexes = np.where(oxidant > 0)[0]
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

        if len(primary_comb_indexes) > 0:
            self.nucleation_probability = (primary_active / self.cells_per_axis**2)/0.13

            self.precip_step(primary_comb_indexes, self.primary_oxidant, self.primary_active,
                             self.primary_product, self.secondary_product)

        if len(secondary_comb_indexes) > 0:
            self.nucleation_probability = 1 - ((primary_active / self.cells_per_axis ** 2) / 0.13)
            self.precip_step(secondary_comb_indexes, self.primary_oxidant, self.secondary_active,
                             self.secondary_product, self.primary_product)

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
            self.precip_step(primary_comb_indexes, self.primary_oxidant, self.primary_active,
                             self.primary_product, self.cumul_product)

        if len(secondary_comb_indexes) > 0:
            self.precip_step(secondary_comb_indexes, self.primary_oxidant, self.secondary_active,
                             self.secondary_product, self.cumul_product)

        if len(ternary_comb_indexes) > 0:
            self.precip_step(ternary_comb_indexes, self.secondary_oxidant, self.primary_active,
                             self.ternary_product, self.cumul_product)

        if len(quaternary_comb_indexes) > 0:
            self.precip_step(quaternary_comb_indexes, self.secondary_oxidant, self.secondary_active,
                             self.quaternary_product, self.cumul_product)

        self.primary_oxidant.transform_to_descards()
        self.secondary_oxidant.transform_to_descards()

    def precip_step(self, indexes, oxidant, active, product, to_check_with=None):
        """
        Precipitation formation function.
        """
        def check_intersections_single_int(seeds):
            """
            """
            all_arounds = self.utils.calc_sur_ind_formation(seeds, active.c3d.shape[2] - 1)
            neighbours = np.array([[active.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
                                   for seed_arrounds in all_arounds], dtype=bool)
            arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
            temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

            # if len(temp_ind) > 0:
            #     seeds = seeds[temp_ind]
            #     neighbours = neighbours[temp_ind]
            #     all_arounds = all_arounds[temp_ind]
            #     flat_arounds = all_arounds[:, 0:6]
            #     flat_neighbours = np.array(
            #         [[self.precipitations3d_init[point[0], point[1], point[2]] for point in seed_arrounds]
            #          for seed_arrounds in flat_arounds], dtype=bool)
            #     arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)
            #     homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            #     needed_prob = self.const_a * 2.718281828 ** (self.const_b * arr_len_in_flat)
            #     needed_prob[homogeneous_ind] = self.nucleation_probability
            #     randomise = np.random.random_sample(len(arr_len_in_flat))
            #     temp_ind = np.where(randomise < needed_prob)[0]

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
                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                exists = [product.full_c3d[point[0], point[1], point[2]] for point in coord]
                temp_ind = np.where(exists)[0]
                coord = np.delete(coord, temp_ind, 0)
                seeds = np.delete(seeds, temp_ind, 0)

                if to_check_with is not None:
                    # to_check_min_self = np.array(self.cumul_product - product.c3d, dtype=np.ubyte)
                    exists = np.array([to_check_with.c3d[point[0], point[1], point[2]] for point in coord], dtype=np.ubyte)
                    # exists = np.array([to_check_min_self[point[0], point[1], point[2]] for point in coord],
                    #                   dtype=np.ubyte)
                    temp_ind = np.where(exists > 0)[0]
                    coord = np.delete(coord, temp_ind, 0)
                    seeds = np.delete(seeds, temp_ind, 0)

                coord = coord.transpose()
                seeds = seeds.transpose()

                active.c3d[coord[0], coord[1], coord[2]] -= 1
                oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                product.c3d[coord[0], coord[1], coord[2]] += 1
                product.fix_full_cells(coord)
                # self.cumul_product[coord[0], coord[1], coord[2]] += 1

        def check_intersections_mult_int(seeds):
            """
            Check intersections between the seeds neighbourhood and the coordinates of inward particles only.
            Compute which seed will become a precipitation and which inward particles should be deleted
            according to threshold_inward conditions. This is a simplified version of the check_intersection() function
            where threshold_outward is equal to 1, so there is no need to check intersection with OUT arrays!

            :param seeds: array of seeds coordinates
            """
            all_arounds = self.utils.calc_sur_ind_formation(seeds, active.c3d.shape[2] - 1)
            neighbours = np.array([[active.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
                                   for seed_arrounds in all_arounds], dtype=bool)
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
                exists = [product.full_c3d[point[0], point[1], point[2]] for point in coord]
                temp_ind = np.where(exists)[0]
                coord = np.delete(coord, temp_ind, 0)
                seeds = np.delete(seeds, temp_ind, 0)

                if to_check_with is not None:
                    exists = [to_check_with.c3d[point[0], point[1], point[2]] for point in coord]
                    temp_ind = np.where(exists)[0]
                    coord = np.delete(coord, temp_ind, 0)
                    seeds = np.delete(seeds, temp_ind, 0)

                if len(seeds) > 0:
                    self_all_arounds = self.utils.calc_sur_ind_formation_noz(seeds, oxidant.c3d.shape[2] - 1)
                    self_neighbours = np.array([[oxidant.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
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
                        to_del_in = np.array([in_to_del[1, indx:indx + self.threshold_inward - 1] for indx in in_start_seed_index],
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

                        active.c3d[coord[0], coord[1], coord[2]] -= 1
                        product.c3d[coord[0], coord[1], coord[2]] += 1
                        product.fix_full_cells(coord)

                        oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1
                        oxidant.c3d[coord_in[0], coord_in[1], coord_in[2]] -= 1

        # def sort_out_in_precip_int(fetch_i, plane_x_ind):
        #     seeds = oxidant.c3d[fetch_i[0], fetch_i[1], plane_x_ind]
        #     seeds = fetch_i[:, np.nonzero(seeds)[0]]
        #
        #     if len(seeds[0]) != 0:
        #         seeds = np.vstack((seeds, np.full(len(seeds[0]), plane_x_ind, dtype=np.short)))
        #         seeds = seeds.transpose()
        #
        #         if self.threshold_outward < 2:
        #             check_intersections_single_int(seeds)

        for p_ind, plane_index in enumerate(reversed(indexes)):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                # randomise = np.random.random_sample(len(oxidant_cells[0]))
                # temp_ind = np.where(randomise < self.nucleation_probability[p_ind])[0]
                # oxidant_cells = oxidant_cells[:, temp_ind]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                    # temp_ind = np.where(in_gb)[0]
                    # oxidant_cells = oxidant_cells[temp_ind]

                    if self.threshold_inward < 2 and len(oxidant_cells) != 0:
                        check_intersections_single_int(oxidant_cells)
                    else:
                        check_intersections_mult_int(oxidant_cells)

    def precip_step_single_int(self, indexes, oxidant, active, product, to_check_with=None, nucleation_probability=1):

        for plane_index in reversed(indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                randomise = np.random.random_sample(len(oxidant_cells[0]))
                temp_ind = np.where(randomise < nucleation_probability)[0]
                oxidant_cells = oxidant_cells[:, temp_ind]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                    # temp_ind = np.where(in_gb)[0]
                    # oxidant_cells = oxidant_cells[temp_ind]

                    # if self.threshold_inward < 2 and len(oxidant_cells) != 0:
                    #     check_intersections_single_int(oxidant_cells)
                    # else:
                    #     check_intersections_mult_int(oxidant_cells)

                    all_arounds = self.utils.calc_sur_ind_formation(oxidant_cells, active.c3d.shape[2] - 1)
                    neighbours = np.array([[active.c3d[point[0], point[1], point[2]] for point in seed_arrounds]
                                           for seed_arrounds in all_arounds], dtype=bool)
                    arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
                    temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

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
                    #     needed_prob[homogeneous_ind] = self.nucleation_probability
                    #     randomise = np.random.random_sample(len(arr_len_in_flat))
                    #     temp_ind = np.where(randomise < needed_prob)[0]

                    if len(temp_ind) > 0:
                        oxidant_cells = oxidant_cells[temp_ind]
                        neighbours = neighbours[temp_ind]
                        all_arounds = all_arounds[temp_ind]
                        in_to_del = np.array(np.nonzero(neighbours))
                        start_seed_index = np.unique(in_to_del[0], return_index=True)[1]
                        to_del = np.array(
                            [in_to_del[1, indx:indx + self.threshold_outward] for indx in start_seed_index],
                            dtype=int)
                        coord = np.array(
                            [all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                            dtype=np.short)
                        # coord = np.reshape(coord, (len(coord) * self.threshold_inward, 3)).transpose()

                        coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                        exists = [product.full_c3d[point[0], point[1], point[2]] for point in coord]
                        temp_ind = np.where(exists)[0]
                        coord = np.delete(coord, temp_ind, 0)
                        oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                        if to_check_with is not None:
                            exists = np.array([to_check_with.c3d[point[0], point[1], point[2]] for point in coord],
                                              dtype=np.ubyte)
                            temp_ind = np.where(exists > 0)[0]
                            coord = np.delete(coord, temp_ind, 0)
                            oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                        coord = coord.transpose()
                        oxidant_cells = oxidant_cells.transpose()

                        active.c3d[coord[0], coord[1], coord[2]] -= 1
                        product.c3d[coord[0], coord[1], coord[2]] += 1
                        product.fix_full_cells(coord)

                        oxidant.c3d[oxidant_cells[0], oxidant_cells[1], oxidant_cells[2]] -= 1

    def diffusion_inward(self):
        self.primary_oxidant.diffuse()
        if self.param["secondary_oxidant_exists"]:
            self.secondary_oxidant.diffuse()

    def diffusion_outward(self):
        if (self.iteration + 1) % self.param["stride"] == 0 and self.iteration != 0:
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

    def calc_precipitation_front_only_cells(self, iteration):
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
                sqr_time = ((iteration + 1) * self.utils.param["sim_time"] / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def save_results(self, iteration):
        if self.param["stride"] > self.param["n_iterations"]:
            self.primary_active.transform_to_descards()
            if self.param["secondary_active_element_exists"]:
                self.secondary_active.transform_to_descards()

        if self.param["inward_diffusion"]:
            self.utils.db.insert_particle_data("primary_oxidant", iteration, self.primary_oxidant.cells)
            if self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_oxidant", iteration, self.secondary_oxidant.cells)

        if self.param["outward_diffusion"]:
            self.utils.db.insert_particle_data("primary_active", iteration, self.primary_active.cells)
            if self.param["secondary_active_element_exists"]:
                self.utils.db.insert_particle_data("secondary_active", iteration, self.secondary_active.cells)

        if self.param["compute_precipitations"]:
            self.utils.db.insert_particle_data("primary_product", iteration, self.primary_product.transform_c3d())

            if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_product", iteration,
                                                   self.secondary_product.transform_c3d())
                self.utils.db.insert_particle_data("ternary_product", iteration,
                                                   self.ternary_product.transform_c3d())
                self.utils.db.insert_particle_data("quaternary_product", iteration,
                                                   self.quaternary_product.transform_c3d())

            elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                self.utils.db.insert_particle_data("secondary_product", iteration, self.secondary_product.transform_c3d())

        if self.param["stride"] > self.param["n_iterations"]:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            if self.param["secondary_active_element_exists"]:
                self.secondary_active.transform_to_3d(self.curr_max_furthest)

    def fix_init_precip(self, u_bound, product, shape):
        self.precipitations3d_init = np.full(shape, False)
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        current_precip = np.array(product.c3d[:, :, 0:u_bound + 2], dtype=np.ubyte)
        current_precip = np.array(np.nonzero(current_precip), dtype=np.short)
        if len(current_precip[0]) > 0:
            current_precip[2] += 1
            self.precipitations3d_init[current_precip[0], current_precip[1], current_precip[2]] = True

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
