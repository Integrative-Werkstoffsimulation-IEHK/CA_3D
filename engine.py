import sys

import numpy as np

import utils
import progressbar
from elements import *
import time
import multiprocessing
import ctypes


class MyBufferCoords:
    def __init__(self, reserve, dtype=np.short):
        self.buffer = np.empty((3, reserve), dtype=dtype)
        self.last_in_buffer = 0

    def append_to_buffer(self, data_array):
        self.buffer[:, self.last_in_buffer:self.last_in_buffer + data_array.shape[1]] = data_array
        self.last_in_buffer += data_array.shape[1]

    def copy_to_buffer(self, data_array):
        self.reset_buffer()
        self.append_to_buffer(data_array)

    def get_elem_at_ind(self, indexes):
        return self.buffer[:, indexes]

    def get_elem_instead_ind(self, indexes):
        return np.delete(self.buffer[:, :self.last_in_buffer], indexes, axis=1)

    def update_buffer_at_axis(self, new_data, axis=2):
        self.buffer[axis, :self.last_in_buffer] = new_data

    def get_buffer(self):
        return self.buffer[:, :self.last_in_buffer]

    def reset_buffer(self):
        self.last_in_buffer = 0


class CellularAutomata:
    def __init__(self):
        self.utils = utils.Utils()
        self.utils.generate_param()
        self.utils.create_database()
        self.elapsed_time = 0

        # simulated space parameters
        self.cells_per_axis = Config.N_CELLS_PER_AXIS
        self.cells_per_page = self.cells_per_axis ** 2
        self.matrix_moles_per_page = self.cells_per_page * Config.MATRIX.MOLES_PER_CELL

        self.n_iter = Config.N_ITERATIONS
        self.iteration = None
        self.curr_max_furthest = 0

        self.cases = utils.CaseRef()
        self.cur_case = None

        # setting objects for inward diffusion
        if Config.INWARD_DIFFUSION:
            self.primary_oxidant = OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)
            self.cases.first.oxidant = self.primary_oxidant
            self.cases.second.oxidant = self.primary_oxidant
            # ---------------------------------------------------
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.secondary_oxidant = OxidantElem(Config.OXIDANTS.SECONDARY, self.utils)
                self.cases.third.oxidant = self.secondary_oxidant
                self.cases.fourth.oxidant = self.secondary_oxidant
        # setting objects for outward diffusion
        if Config.OUTWARD_DIFFUSION:
            self.primary_active = ActiveElem(Config.ACTIVES.PRIMARY)
            self.cases.first.active = self.primary_active
            self.cases.third.active = self.primary_active
            # ---------------------------------------------------
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.secondary_active = ActiveElem(Config.ACTIVES.SECONDARY)
                self.cases.second.active = self.secondary_active
                self.cases.fourth.active = self.secondary_active
                # ---------------------------------------------------
        # setting objects for precipitations
        if Config.COMPUTE_PRECIPITATION:
            self.precip_func = None  # must be defined elsewhere
            self.get_combi_ind = None  # must be defined elsewhere
            self.precip_step = None  # must be defined elsewhere
            self.check_intersection = None  # must be defined elsewhere
            self.decomposition = None  # must be defined elsewhere
            self.decomposition_intrinsic = None  # must be defined elsewhere

            self.coord_buffer = utils.my_data_structs.MyBufferCoords(Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER * self.cells_per_axis ** 3)
            self.to_dissol_pn_buffer = utils.my_data_structs.MyBufferCoords(Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER * self.cells_per_axis ** 3)

            self.primary_product = Product(Config.PRODUCTS.PRIMARY)
            self.cases.first.product = self.primary_product

            self.primary_oxid_numb = Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER
            self.max_inside_neigh_number = 6 * self.primary_oxid_numb
            # self.max_block_neigh_number = 7 * self.primary_oxid_numb
            self.max_block_neigh_number = 7

            self.disol_block_p = Config.PROBABILITIES.PRIMARY.p0_d ** Config.PROBABILITIES.PRIMARY.n
            self.disol_p = Config.PROBABILITIES.PRIMARY.p0_d

            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.secondary_product = Product(self.param["product"]["secondary"])
                self.objs[1]["product"] = self.secondary_product
                self.ternary_product = Product(self.param["product"]["ternary"])
                self.objs[2]["product"] = self.ternary_product
                self.quaternary_product = Product(self.param["product"]["quaternary"])
                self.objs[3]["product"] = self.quaternary_product
                self.objs[0]["to_check_with"] = self.cumul_product
                self.objs[1]["to_check_with"] = self.cumul_product
                self.objs[2]["to_check_with"] = self.cumul_product
                self.objs[3]["to_check_with"] = self.cumul_product

            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.secondary_product = Product(Config.PRODUCTS.SECONDARY)
                self.cases.second.product = self.secondary_product
                self.cases.first.to_check_with = self.secondary_product
                self.cases.second.to_check_with = self.primary_product

                self.cases.first.prod_indexes = np.full(self.cells_per_axis, False, dtype=bool)
                self.cases.second.prod_indexes = np.full(self.cells_per_axis, False, dtype=bool)

                if self.cases.first.product.oxidation_number == 1:
                    self.cases.first.go_around_func_ref = self.go_around_single_oxid_n
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_bool
                    my_type = bool
                else:
                    self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_int
                    my_type = np.ubyte
                self.cases.first.precip_3d_init = np.full(
                    (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1),
                    0, dtype=my_type)

                if self.cases.second.product.oxidation_number == 1:
                    self.cases.second.go_around_func_ref = self.go_around_single_oxid_n
                    self.cases.second.fix_init_precip_func_ref = self.fix_init_precip_bool
                    my_type = bool
                else:
                    self.cases.second.go_around_func_ref = self.go_around_mult_oxid_n
                    self.cases.second.fix_init_precip_func_ref = self.fix_init_precip_int
                    my_type = np.ubyte
                self.cases.second.precip_3d_init = np.full(
                    (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1),
                    0, dtype=my_type)
            else:
                self.primary_oxidant.scale = self.primary_product
                self.primary_active.scale = self.primary_product

                if self.cases.first.product.oxidation_number == 1:
                    self.cases.first.go_around_func_ref = self.go_around_single_oxid_n
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_bool
                    self.cases.first.precip_3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1), False, dtype=bool)
                else:
                    self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n
                    # self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n_also_partial_neigh  # CHANGE!!!!
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_int
                    self.cases.first.precip_3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1), 0, dtype=np.ubyte)

                # # Create shared memory array
                self.shared_array_base = multiprocessing.Array(ctypes.c_ubyte, self.primary_product.c3d.size, lock=False)
                self.shared_array = np.frombuffer(self.shared_array_base, dtype=ctypes.c_ubyte).reshape(self.primary_product.c3d.shape)

                # self.shared_array[:] = self.primary_product.c3d.flatten()

                # self.result_array_base = multiprocessing.Array(ctypes.c_ubyte, self.primary_product.c3d.size, lock=False)
                # self.result_array = np.frombuffer(self.result_array_base, dtype=np.ubyte).reshape(self.primary_product.c3d.shape)

                self.numb_of_proc = 8
                chunk_size = int((self.cells_per_axis / (self.numb_of_proc - 1)))

                self.chunk_ranges = np.zeros((self.numb_of_proc, 2), dtype=int)
                for pos, item in enumerate(np.arange(0, self.cells_per_axis, chunk_size)):
                    self.chunk_ranges[pos] = [item, item+chunk_size]
                self.chunk_ranges[-1, -1] = self.cells_per_axis

                # self.pool = multiprocessing.Pool(processes=self.numb_of_proc)
                self.buffer_size = Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER * self.cells_per_axis * chunk_size

            self.threshold_inward = Config.THRESHOLD_INWARD
            self.threshold_outward = Config.THRESHOLD_OUTWARD

            self.fetch_ind = None
            self.generate_fetch_ind()

            # self.primary_product.c3d[1, 1, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[1, 1, 50] = True
            #
            # self.primary_product.c3d[1, 1, 49] = self.primary_oxid_numb
            # self.primary_product.full_c3d[1, 1, 49] = True
            #
            # self.primary_product.c3d[1, 2, 49] = self.primary_oxid_numb
            # self.primary_product.full_c3d[1, 2, 49] = True
            #
            # self.primary_product.c3d[1, 2, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[1, 2, 50] = True
            #
            # self.primary_product.c3d[1, 3, 49] = self.primary_oxid_numb
            # self.primary_product.full_c3d[1, 3, 49] = True
            #
            #
            # self.primary_product.c3d[2, 1, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 1, 50] = True
            #
            # self.primary_product.c3d[2, 1, 49] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 1, 49] = True
            #
            # self.primary_product.c3d[2, 2, 49] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 2, 49] = True
            #
            # self.primary_product.c3d[2, 2, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 2, 50] = True
            #
            #
            # self.primary_product.c3d[2, 0, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 0, 50] = True
            #
            # self.primary_product.c3d[2, 1, 51] = self.primary_oxid_numb
            # self.primary_product.full_c3d[2, 1, 51] = True
            #
            # self.primary_product.c3d[3, 1, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[3, 1, 50] = True
            #
            #
            # self.primary_product.c3d[3, 3, 50] = 2
            # self.primary_product.c3d[10, 3, 50] = 2
            # self.primary_product.c3d[35, 3, 50] = 2
            #
            #
            # self.primary_product.c3d[1, 10, 50] = 2
            # self.primary_product.c3d[1, 11, 50] = 2
            # self.primary_product.c3d[1, 9, 50] = 2
            # self.primary_product.c3d[1, 10, 49] = 2
            #
            # self.primary_product.c3d[45, 3, 50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[45, 3, 50] = True

            # self.primary_product.full_c3d[minus:plus, minus:plus, minus:plus] = True
            # self.primary_product.full_c3d[1, 1, 50] = True
            # shift = 0
            # self.precipitations3d[minus + shift:plus + shift, minus + shift:plus + shift, minus + shift:plus + shift] = True
            # self.precipitations = np.array(np.nonzero(self.precipitations), dtype=int)
            # self.precipitations3d = np.full(self.single_page_shape, False)
            # self.precipitations3d_sec = np.full(self.single_page_shape, False)

            # self.primary_product.c3d[:, :, :50] = self.primary_oxid_numb
            # self.primary_product.full_c3d[:, :, :50] = True

            self.aggregated_ind = np.array([[7, 0, 1, 2, 19, 16, 14],
                                   [6, 0, 1, 5, 18, 15, 14],
                                   [8, 0, 4, 5, 20, 15, 17],
                                   [9, 0, 4, 2, 21, 16, 17],
                                   [11, 3, 1, 2, 19, 24, 22],
                                   [10, 3, 1, 5, 18, 23, 22],
                                   [12, 3, 4, 5, 20, 23, 25],
                                   [13, 3, 4, 2, 21, 24, 25]], dtype=np.int64)

            self.cases.first.nucleation_probabilities = None  # must be defined elsewhere
            self.cases.first.dissolution_probabilities = None  # must be defined elsewhere

            self.cases.second.nucleation_probabilities = None  # must be defined elsewhere
            self.cases.second.dissolution_probabilities = None  # must be defined elsewhere

            self.cases.third.nucleation_probabilities = None  # must be defined elsewhere
            self.cases.third.dissolution_probabilities = None  # must be defined elsewhere

            self.cases.fourth.nucleation_probabilities = None  # must be defined elsewhere
            self.cases.fourth.dissolution_probabilities = None  # must be defined elsewhere

            self.furthest_index = 0
            self.comb_indexes = None
            self.rel_prod_fraction = None
            self.gamma_primes = None
            self.product_indexes = None
            self.nucleation_indexes = None

            self.save_flag = False
            self.product_x_nzs = np.full(self.cells_per_axis, False, dtype=bool)
            self.product_x_not_stab = np.full(self.cells_per_axis, True, dtype=bool)
            # self.TdDATA = TdDATA()
            # self.TdDATA.fetch_look_up_from_file()
            self.curr_look_up = None

            self.prev_stab_count = 0

            self.precipitation_stride = Config.STRIDE * Config.STRIDE_MULTIPLIER

            self.cumul_prod = utils.my_data_structs.MyBufferSingle(self.n_iter, dtype=float)
            self.growth_rate = utils.my_data_structs.MyBufferSingle(self.n_iter, dtype=float)

            self.cumul_prod1 = utils.my_data_structs.MyBufferSingle(self.n_iter, dtype=float)
            self.growth_rate1 = utils.my_data_structs.MyBufferSingle(self.n_iter, dtype=float)

            # self.soll_prod = 0
            self.diffs = None
            self.curr_time = 0

            lambdas = (np.arange(self.cells_per_axis, dtype=int) + 0.5) * Config.GENERATED_VALUES.LAMBDA
            adj_lamd = lambdas - Config.ZETTA_ZERO
            neg_ind = np.where(adj_lamd < 0)[0]
            adj_lamd[neg_ind] = 0
            self.active_times = adj_lamd ** 2 / Config.GENERATED_VALUES.KINETIC_KONST ** 2

        self.begin = time.time()

        self.ind_decompose_no_flat = np.array(
            [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_flat_z = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]], dtype=np.byte)

    def simulation(self):
        for self.iteration in progressbar.progressbar(range(self.n_iter)):
            # if self.iteration % self.precipitation_stride == 0:
            self.precip_func()
            self.decomposition()
                # self.calc_precipitation_front_only_cells()
            # self.precip_func()
            # self.decomposition()
            self.diffusion_inward()
            self.diffusion_outward()
            # self.diffusion_outward_with_mult_srtide()
            # self.save_results()

        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.utils.db.insert_time(self.elapsed_time)
        self.utils.db.conn.commit()

    def dissolution_zhou_wei_original(self):
        """Implementation of original not adapted Zhou and Wei approach. Only two probabilities p for block and pn
        are considered. Works for any oxidation nuber!"""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            # all_neigh_pn = all_neigh[[]]
            # all_neigh_block = all_neigh[[]]

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
                    # all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    # all_neigh_pn = all_neigh

                    self.coord_buffer.reset_buffer()
                    # all_neigh_block = all_neigh[[]]

            # probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]
            probs_pn_no_neigh = np.full(len(to_dissol_pn_no_neigh[0]), self.disol_p)

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            # all_neigh_pn = np.array([np.sum(item[:6]) for item in all_neigh_pn])
            # all_neigh_pn = np.zeros(len(all_neigh_pn))
            # probs_pn = self.dissol_prob.dissol_prob.values_pp[to_dissolve_pn[2]]
            probs_pn = np.full(len(to_dissolve_pn[0]), self.disol_p)

            to_dissolve_p = self.coord_buffer.get_buffer()
            # all_neigh_block = np.array([np.sum(item[:6]) for item in all_neigh_block])
            # all_neigh_block = np.full(len(all_neigh_block), self.primary_oxid_numb * 3)
            # probs_p = self.dissol_prob.get_probabilities_block(all_neigh_block, to_dissolve_p[2])
            probs_p = np.full(len(to_dissolve_p[0]), self.disol_block_p)

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

    def dissolution_zhou_wei_with_bsf(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!"""
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            all_neigh_pn = all_neigh[[]]
            all_neigh_block = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh[:, :6].view() == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))

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

            probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

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

    def dissolution_zhou_wei_with_bsf_aip(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!"""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)

                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                non_flat_neigh = go_around_int(self.primary_product.c3d, non_flat_arounds)
                in_prod_column = np.array([all_neigh[:, 6]]).transpose()
                all_neigh = np.concatenate((all_neigh[:, :6], non_flat_neigh, in_prod_column), axis=1)
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh, dtype=bool)

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)

                # if len(ind_where_blocks) > 0:
                #
                #     begin = time.time()
                #     aggregation = np.array(
                #         [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #         dtype=np.ubyte)
                #     ind_where_blocks = np.unique(np.where(aggregation == 7)[0])
                #     print("list comp: ", time.time() - begin)
                #
                #     begin = time.time()
                #     ind_where_blocks2 = aggregate(self.aggregated_ind, all_neigh_bool)
                #     print("numba: ", time.time() - begin)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_no_block = np.delete(all_neigh[:, :6], ind_where_blocks, axis=0)
                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block[:, :6], axis=1) + numb_in_prod_no_block
                    all_neigh_no_block = np.sum(all_neigh_no_block, axis=1) + numb_in_prod_no_block

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                    all_neigh_block = all_neigh[ind_where_blocks, :6]
                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                    # all_neigh_block = np.sum(all_neigh_block[:, :6], axis=1) + numb_in_prod_block
                    all_neigh_block = np.sum(all_neigh_block, axis=1) + numb_in_prod_block
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_no_block = np.sum(all_neigh[:, :6], axis=1) + numb_in_prod
                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block, to_dissolve_no_block[2])
            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf_aip_UPGRADE(self):
        """Implementation of Zhou and Wei approach. Works for any oxidation nuber!
        Here the problem was that the geometrical arrangement is considered properly! For higher oxidation numbers the
         total number of neighbours does correlate with the geometrical configuration of the cluster!! 3 neighbours here
         mean to 3 geometrical flat neighbours."""

        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)

                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh = all_neigh[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                non_flat_neigh = go_around_int(self.primary_product.c3d, non_flat_arounds)
                in_prod_column = np.array([all_neigh[:, 6]]).transpose()
                all_neigh = np.concatenate((all_neigh[:, :6], non_flat_neigh, in_prod_column), axis=1)
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh, dtype=bool)

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)
                # block_counts = aggregate_and_count(self.aggregated_ind, all_neigh_bool)
                # some = np.where(block_counts > 4)[0]
                #
                # if len(some) > 0:
                #     print()
                # ind_where_blocks = np.where(block_counts)[0]

                # if len(ind_where_blocks) > 0:
                #
                #     begin = time.time()
                #     aggregation = np.array(
                #         [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #         dtype=np.ubyte)
                #     ind_where_blocks = np.unique(np.where(aggregation == 7)[0])
                #     print("list comp: ", time.time() - begin)
                #
                #     begin = time.time()
                #     ind_where_blocks2 = aggregate(self.aggregated_ind, all_neigh_bool)
                #     print("numba: ", time.time() - begin)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    all_neigh_no_block = np.delete(all_neigh[:, :6], ind_where_blocks, axis=0)

                    all_neigh_bool = np.delete(all_neigh_bool[:, :6], ind_where_blocks, axis=0)
                    all_neigh_bool = np.sum(all_neigh_bool, axis=1)
                    ind_to_raise = np.where((all_neigh_bool == 3) | (all_neigh_bool == 4))[0]

                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block[:, :6], axis=1) + numb_in_prod_no_block
                    all_neigh_no_block = np.sum(all_neigh_no_block, axis=1) + numb_in_prod_no_block

                    all_neigh_no_block[ind_to_raise] = 0

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                    all_neigh_block = all_neigh[ind_where_blocks, :6]
                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                    # all_neigh_block = np.sum(all_neigh_block[:, :6], axis=1) + numb_in_prod_block
                    all_neigh_block = np.sum(all_neigh_block, axis=1) + numb_in_prod_block
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    all_neigh_no_block = np.sum(all_neigh[:, :6], axis=1) + numb_in_prod

                    all_neigh_bool = np.sum(all_neigh_bool[:, :6], axis=1)
                    ind_to_raise = np.where((all_neigh_bool == 3) | (all_neigh_bool == 4))[0]

                    all_neigh_no_block[ind_to_raise] = 0

                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block,
                                                                                       to_dissolve_no_block[2])
            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block,
                                                                                          to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL(self):
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            flat_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, flat_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_block = np.array([])
            all_neigh_no_block = np.array([])
            numb_in_prod_block = np.array([], dtype=int)
            numb_in_prod_no_block = np.array([], dtype=int)
            ind_to_raise = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
            to_dissol_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                numb_in_prod = all_neigh[:, -1]

                all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)

                arr_len_flat = np.sum(all_neigh_bool, axis=1)

                index_outside = np.where((arr_len_flat < 6))[0]
                self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))

                # all_neigh = all_neigh[index_outside]
                all_neigh_bool = all_neigh_bool[index_outside]
                arr_len_flat = arr_len_flat[index_outside]
                numb_in_prod = numb_in_prod[index_outside]

                non_flat_arounds = self.utils.calc_sur_ind_decompose_no_flat(self.coord_buffer.get_buffer())
                # non_flat_neigh = go_around_int(self.primary_product.c3d, non_flat_arounds)
                non_flat_neigh = go_around_bool(self.primary_product.c3d, non_flat_arounds)
                # in_prod_column = np.array([all_neigh[:, 6]]).transpose()
                # all_neigh = np.concatenate((all_neigh[:, :6], non_flat_neigh, in_prod_column), axis=1)
                all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)

                # all_neigh_bool = np.array(all_neigh, dtype=bool)

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == 7)[0])

                ind_where_blocks = aggregate(self.aggregated_ind, all_neigh_bool)
                # block_counts = aggregate_and_count(self.aggregated_ind, all_neigh_bool)
                # some = np.where(block_counts > 4)[0]
                #
                # if len(some) > 0:
                #     print()
                # ind_where_blocks = np.where(block_counts)[0]

                # if len(ind_where_blocks) > 0:
                #
                #     begin = time.time()
                #     aggregation = np.array(
                #         [[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh_bool],
                #         dtype=np.ubyte)
                #     ind_where_blocks = np.unique(np.where(aggregation == 7)[0])
                #     print("list comp: ", time.time() - begin)
                #
                #     begin = time.time()
                #     ind_where_blocks2 = aggregate(self.aggregated_ind, all_neigh_bool)
                #     print("numba: ", time.time() - begin)

                if len(ind_where_blocks) > 0:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                    # all_neigh_no_block = np.delete(all_neigh[:, :6], ind_where_blocks, axis=0)

                    # all_neigh_no_block = np.delete(all_neigh_bool[:, :6], ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block, axis=1)

                    all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)

                    # ind_to_raise = np.where((all_neigh_no_block == 3) | (all_neigh_no_block == 4))[0]

                    numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
                    # all_neigh_no_block = np.sum(all_neigh_no_block[:, :6], axis=1) + numb_in_prod_no_block
                    # all_neigh_no_block = np.sum(all_neigh_no_block, axis=1) + numb_in_prod_no_block

                    # all_neigh_no_block[ind_to_raise] = 0

                    self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                    # all_neigh_block = all_neigh[ind_where_blocks, :6]
                    all_neigh_block = arr_len_flat[ind_where_blocks]

                    numb_in_prod_block = numb_in_prod[ind_where_blocks]
                    # all_neigh_block = np.sum(all_neigh_block[:, :6], axis=1) + numb_in_prod_block
                    # all_neigh_block = np.sum(all_neigh_block, axis=1)
                else:
                    self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    # all_neigh_no_block = np.sum(all_neigh[:, :6], axis=1) + numb_in_prod
                    all_neigh_no_block = arr_len_flat
                    # all_neigh_bool = np.sum(all_neigh_bool[:, :6], axis=1)

                    # ind_to_raise = np.where((all_neigh_no_block == 3) | (all_neigh_no_block == 4))[0]

                    # all_neigh_no_block[ind_to_raise] = 0

                    numb_in_prod_no_block = numb_in_prod

                    self.coord_buffer.reset_buffer()
                    all_neigh_block = np.array([])
                    numb_in_prod_block = np.array([], dtype=int)

            to_dissolve_no_block = self.to_dissol_pn_buffer.get_buffer()
            probs_no_block = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block,
                                                                                       to_dissolve_no_block[2])
            # probs_no_block[ind_to_raise] = 1

            non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
            to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
            probs_no_block = np.concatenate((probs_no_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
            temp_ind = np.where(randomise < probs_no_block)[0]
            to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]

            to_dissolve_block = self.coord_buffer.get_buffer()
            probs_block = self.cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block,
                                                                                          to_dissolve_block[2])
            non_z_ind = np.where(numb_in_prod_block != 0)[0]
            repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
            to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
            probs_block = np.concatenate((probs_block, repeated_probs))
            randomise = np.random.random_sample(len(to_dissolve_block[0]))
            temp_ind = np.where(randomise < probs_block)[0]
            to_dissolve_block = to_dissolve_block[:, temp_ind]

            probs_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
            temp_ind = np.where(randomise < probs_no_neigh)[0]
            to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dummy(self):
        return "asdk"

    def dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL_MP(self, some):
        # , chunk_range, comb_ind, buffer_size, aggregated_ind
        print(some)

        # coord_buffer = MyBufferCoords(1000000)
        # to_dissol_pn_buffer = MyBufferCoords(1000000)
        #
        # chunk_range = [0, 0]
        # comb_ind = [0]
        #
        # nz_ind = np.array(np.nonzero(self.shared_array[chunk_range[0]:chunk_range[1], :, comb_ind]))
        # coord_buffer.copy_to_buffer(nz_ind)
        # coord_buffer.update_buffer_at_axis(comb_ind[nz_ind[2]], axis=2)
        #
        # if coord_buffer.last_in_buffer > 0:
        #     flat_arounds = self.calc_sur_ind_decompose_flat_with_zero(coord_buffer.get_buffer())
        #     all_neigh = go_around_int(self.shared_array, flat_arounds)
        #     all_neigh[:, 6] -= 1
        #
        #     all_neigh_block = np.array([])
        #     all_neigh_no_block = np.array([])
        #     numb_in_prod_block = np.array([], dtype=int)
        #     numb_in_prod_no_block = np.array([], dtype=int)
        #
        #     where_not_null = np.unique(np.where(all_neigh[:, :6] > 0)[0])
        #     to_dissol_no_neigh = np.array(coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
        #     coord_buffer.copy_to_buffer(coord_buffer.get_elem_at_ind(where_not_null))
        #
        #     if coord_buffer.last_in_buffer > 0:
        #         all_neigh = all_neigh[where_not_null]
        #         numb_in_prod = all_neigh[:, -1]
        #
        #         all_neigh_bool = np.array(all_neigh[:, :6], dtype=bool)
        #
        #         arr_len_flat = np.sum(all_neigh_bool, axis=1)
        #
        #         index_outside = np.where((arr_len_flat < 6))[0]
        #         coord_buffer.copy_to_buffer(coord_buffer.get_elem_at_ind(index_outside))
        #
        #         all_neigh_bool = all_neigh_bool[index_outside]
        #         arr_len_flat = arr_len_flat[index_outside]
        #         numb_in_prod = numb_in_prod[index_outside]
        #
        #         non_flat_arounds = self.calc_sur_ind_decompose_no_flat(coord_buffer.get_buffer())
        #         non_flat_neigh = go_around_bool(self.shared_array, non_flat_arounds)
        #         all_neigh_bool = np.concatenate((all_neigh_bool, non_flat_neigh), axis=1)
        #
        #         ind_where_blocks = aggregate(aggregated_ind, all_neigh_bool)
        #
        #         if len(ind_where_blocks) > 0:
        #             to_dissol_pn_buffer.copy_to_buffer(coord_buffer.get_elem_instead_ind(ind_where_blocks))
        #             all_neigh_no_block = np.delete(arr_len_flat, ind_where_blocks)
        #             numb_in_prod_no_block = np.delete(numb_in_prod, ind_where_blocks, axis=0)
        #
        #             coord_buffer.copy_to_buffer(coord_buffer.get_elem_at_ind(ind_where_blocks))
        #             all_neigh_block = arr_len_flat[ind_where_blocks]
        #
        #             numb_in_prod_block = numb_in_prod[ind_where_blocks]
        #         else:
        #             to_dissol_pn_buffer.copy_to_buffer(coord_buffer.get_buffer())
        #             all_neigh_no_block = arr_len_flat
        #             numb_in_prod_no_block = numb_in_prod
        #
        #             coord_buffer.reset_buffer()
        #             all_neigh_block = np.array([])
        #             numb_in_prod_block = np.array([], dtype=int)
        #
        #     to_dissolve_no_block = to_dissol_pn_buffer.get_buffer()
        #     # probs_no_block = cur_case.dissolution_probabilities.get_probabilities(all_neigh_no_block,
        #     #                                                                            to_dissolve_no_block[2])
        #     probs_no_block = np.ones(len(to_dissolve_no_block))
        #
        #     non_z_ind = np.where(numb_in_prod_no_block != 0)[0]
        #     repeated_coords = np.repeat(to_dissolve_no_block[:, non_z_ind], numb_in_prod_no_block[non_z_ind], axis=1)
        #     repeated_probs = np.repeat(probs_no_block[non_z_ind], numb_in_prod_no_block[non_z_ind])
        #     to_dissolve_no_block = np.concatenate((to_dissolve_no_block, repeated_coords), axis=1)
        #     probs_no_block = np.concatenate((probs_no_block, repeated_probs))
        #     randomise = np.random.random_sample(len(to_dissolve_no_block[0]))
        #     temp_ind = np.where(randomise < probs_no_block)[0]
        #     to_dissolve_no_block = to_dissolve_no_block[:, temp_ind]
        #
        #     to_dissolve_block = coord_buffer.get_buffer()
        #     # probs_block = cur_case.dissolution_probabilities.get_probabilities_block(all_neigh_block, to_dissolve_block[2])
        #     probs_block = np.ones(len(to_dissolve_block))
        #
        #     non_z_ind = np.where(numb_in_prod_block != 0)[0]
        #     repeated_coords = np.repeat(to_dissolve_block[:, non_z_ind], numb_in_prod_block[non_z_ind], axis=1)
        #     repeated_probs = np.repeat(probs_block[non_z_ind], numb_in_prod_block[non_z_ind])
        #     to_dissolve_block = np.concatenate((to_dissolve_block, repeated_coords), axis=1)
        #     probs_block = np.concatenate((probs_block, repeated_probs))
        #     randomise = np.random.random_sample(len(to_dissolve_block[0]))
        #     temp_ind = np.where(randomise < probs_block)[0]
        #     to_dissolve_block = to_dissolve_block[:, temp_ind]
        #
        #     # probs_no_neigh = cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_no_neigh[2]]
        #     probs_no_neigh = np.ones(len(to_dissol_no_neigh))
        #     randomise = np.random.random_sample(len(to_dissol_no_neigh[0]))
        #     temp_ind = np.where(randomise < probs_no_neigh)[0]
        #     to_dissol_no_neigh = to_dissol_no_neigh[:, temp_ind]
        #
        #     to_dissolve = np.concatenate((to_dissolve_no_block, to_dissol_no_neigh, to_dissolve_block), axis=1)
        #     return to_dissolve

    def calc_sur_ind_decompose_flat_with_zero(self, seeds):
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_flat_z] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_decompose_no_flat(self, seeds):
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_no_flat] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.cells_per_axis - 1
        return around_seeds

    def dissolution_zhou_wei_no_bsf(self):
        """
        Implementation of adjusted Zhou and Wei approach. Only side neighbours are checked. No need for block scale
        factor. Works only for any oxidation nuber!
        """
        nz_ind = np.array(np.nonzero(self.cur_case.product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose_flat(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            all_neigh_pn = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_full]

                arr_len_flat = np.array([np.sum(item) for item in all_neigh], dtype=np.ubyte)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh_pn = arr_len_flat[index_outside]
            else:
                all_neigh_pn = np.array([np.sum(item) for item in all_neigh_pn])

            probs_pn_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                counts = self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]]
                self.primary_product.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = 0
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] += counts
                # self.primary_active.c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]+1] += counts
                to_dissolve = np.repeat(to_dissolve, counts, axis=1)
                # to_dissolve[2, :] -= 1
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_zhou_wei_no_bsf_also_partial_neigh_aip(self):
        """
        Implementation of adjusted Zhou and Wei approach. Only side neighbours are checked. No need for block scale
        factor. Works for oxidation nuber > 1!
        aip: Adjusted Inside Product!
        Im Gegensatz zu dissolution_zhou_wei_no_bsf werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!
        Resolution inside a product: probability for each partial product adjusted according to a number of neighbours
        """
        nz_ind = np.array(np.nonzero(self.cur_case.product.c3d[:, :, self.comb_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.comb_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose_flat_with_zero(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)
            all_neigh[:, 6] -= 1

            all_neigh_pn = np.array([])
            numb_in_prod = np.array([], dtype=int)

            where_not_null = np.unique(np.where(all_neigh > 0)[0])
            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_not_null), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_not_null))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_not_null]
                numb_in_prod = all_neigh[:, 6]

                arr_len_flat = np.sum(all_neigh[:, :6], axis=1)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh_pn = arr_len_flat[index_outside]
                numb_in_prod = np.array(numb_in_prod[index_outside], dtype=int)

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.cur_case.dissolution_probabilities.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            non_z_ind = np.where(numb_in_prod != 0)[0]
            repeated_coords = np.repeat(to_dissolve_pn[:, non_z_ind], numb_in_prod[non_z_ind], axis=1)
            repeated_probs = np.repeat(probs_pn[non_z_ind], numb_in_prod[non_z_ind])

            to_dissolve_pn = np.concatenate((to_dissolve_pn, repeated_coords), axis=1)
            probs_pn = np.concatenate((probs_pn, repeated_probs))

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            probs_pn_no_neigh = self.cur_case.dissolution_probabilities.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]
            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            to_dissolve = np.concatenate((to_dissolve_pn, to_dissol_pn_no_neigh), axis=1)

            self.coord_buffer.reset_buffer()
            self.to_dissol_pn_buffer.reset_buffer()

            if len(to_dissolve[0]) > 0:
                just_decrease_counts(self.primary_product.c3d, to_dissolve)
                self.primary_product.full_c3d[to_dissolve[0], to_dissolve[1], to_dissolve[2]] = False
                insert_counts(self.primary_active.c3d, to_dissolve)
                self.primary_oxidant.cells = np.concatenate((self.primary_oxidant.cells, to_dissolve), axis=1)
                new_dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(to_dissolve[0]))
                new_dirs = np.array(np.unravel_index(new_dirs, (3, 3, 3)), dtype=np.byte)
                new_dirs -= 1
                self.primary_oxidant.dirs = np.concatenate((self.primary_oxidant.dirs, new_dirs), axis=1)

    def dissolution_simple_with_pd(self):
        """
        Implementation of a simple dissolution approach with single pd for dissolution. No side neighbours are checked,
        no block scale factor, no p_block.
        Works only for any oxidation nuber!
        """
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            to_dissolve = self.coord_buffer.get_buffer()
            probs = np.full(len(to_dissolve[0]), self.disol_p)

            randomise = np.random.random_sample(len(to_dissolve[0]))
            temp_ind = np.where(randomise < probs)[0]
            to_dissolve = to_dissolve[:, temp_ind]

            self.coord_buffer.reset_buffer()

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

    def get_combi_ind_standard(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]

        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = [self.furthest_index]

    def get_combi_ind_cells_around_product(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)

        # self.product_indexes = np.where((product_c < self.param["phase_fraction_lim"]) & (product_c > 0))[0]
        self.product_indexes = np.where(product > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        # some = np.where((product_c[self.product_indexes] < self.param["phase_fraction_lim"]) & (product_c[self.product_indexes] > 0))[0]
        # some = np.where(product_c[self.product_indexes] < self.param["phase_fraction_lim"])[0]
        # self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

    def get_combi_ind_atomic_gamma_prime(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles
        product_c = product_moles / whole_moles

        self.gamma_primes = ((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /\
                            Config.GENERATED_VALUES.max_gamma_min_one

        where_solub_prod = np.where(self.gamma_primes > 0)[0]
        temp_ind = np.where(product_c[where_solub_prod] < Config.PHASE_FRACTION_LIMIT)[0]
        where_solub_prod = where_solub_prod[temp_ind]

        self.rel_prod_fraction = product_c / Config.PHASE_FRACTION_LIMIT

        self.product_indexes = np.where(product_c > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        some = np.where(product_c[self.product_indexes] < Config.PHASE_FRACTION_LIMIT)[0]
        self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, where_solub_prod)))

    def get_combi_ind_atomic(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        product_c = product_moles / whole_moles
        self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_no_growth(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        product_c = product_moles / whole_moles
        self.product_indexes = np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

    def get_combi_ind_two_products(self, current_active):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        active = np.array([np.sum(current_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)

    def get_combi_ind_atomic_two_products(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -\
                       secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles +\
                      secondary_active_moles + secondary_product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles
        secondary_active_c = secondary_active_moles / whole_moles
        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        self.gamma_primes = (((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /
                             Config.GENERATED_VALUES.max_gamma_min_one)

        where_solub_prod = np.where(self.gamma_primes > 0)[0]
        temp_ind = np.where(product_c[where_solub_prod] < Config.PHASE_FRACTION_LIMIT)[0]
        where_solub_prod = where_solub_prod[temp_ind]

        self.rel_prod_fraction = product_c / Config.PHASE_FRACTION_LIMIT

        self.product_indexes = np.where(product_c > 0)[0]
        prod_left_shift = self.product_indexes - 1
        prod_right_shift = self.product_indexes + 1
        self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        self.product_indexes = self.product_indexes[temp_ind]

        some = np.where(product_c[self.product_indexes] < Config.PHASE_FRACTION_LIMIT)[0]
        self.product_indexes = self.product_indexes[some]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            comb_indexes = oxidant_indexes[indexs]
            # self.comb_indexes = comb_indexes
            self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, where_solub_prod)))

    def get_combi_ind_atomic_with_kinetic(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        self.soll_prod = Config.PROD_INCR_CONST * (Config.GENERATED_VALUES.TAU * (self.iteration + 1))**1.1

        self.cumul_prod.append(product_c[0])
        self.growth_rate.append(self.soll_prod)

        self.product_indexes = np.where((product_c <= Config.PHASE_FRACTION_LIMIT) & (product_c < self.soll_prod))[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_with_kinetic_and_KP(self):
        self.curr_time = Config.GENERATED_VALUES.TAU * (self.iteration + 1)

        active_ind = np.where(self.active_times <= self.curr_time)[0]
        # if self.furthest_index > np.amax(active_ind):
        #     print("Not Stable!")
        ioz_bound = min(np.amax(active_ind), self.furthest_index)
        # fur = self.furthest_index

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(ioz_bound + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(ioz_bound + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(ioz_bound + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        # self.soll_prod = Config.PROD_INCR_CONST * (Config.GENERATED_VALUES.TAU * (self.iteration + 1))**1.1

        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[:ioz_bound + 1])**1.1
        self.diffs = product_c - soll_prod

        self.cumul_prod.append(product_c[0])
        self.growth_rate.append(soll_prod[0])

        if ioz_bound >= 10:
            self.cumul_prod1.append(product_c[10])
            self.growth_rate1.append(soll_prod[10])
        else:
            self.cumul_prod1.append(0)
            self.growth_rate1.append(0)

        # self.product_indexes = np.where((product_c <= Config.PHASE_FRACTION_LIMIT) & (product_c < self.soll_prod))[0]

        self.product_indexes = np.where((product_c <= Config.PHASE_FRACTION_LIMIT) & (self.diffs <= 0))[0]

        self.comb_indexes = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(self.comb_indexes, self.product_indexes)

    def get_combi_ind_atomic_solub_prod_test(self):
        """
        Created only for tests of the solubility product probability function
        """
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles

        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles

        oxidant_c = oxidant_moles / whole_moles
        active_c = active_moles / whole_moles

        self.gamma_primes = (((((oxidant_c ** 3) * (active_c ** 2)) / Config.SOL_PROD) - 1) /
                             Config.GENERATED_VALUES.max_gamma_min_one)

        where_solub_prod = np.where(self.gamma_primes > 0)[0]

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = [self.furthest_index]

        self.comb_indexes = np.intersect1d(self.comb_indexes, where_solub_prod)

    def get_combi_ind_atomic_opt_for_growth(self):

        w_int = np.where(self.product_x_not_stab[:self.furthest_index + 1])[0]

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind in w_int], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        self.nucleation_indexes = w_int[np.where(product_c <= Config.PHASE_FRACTION_LIMIT)[0]]

        stab_prod_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        self.product_x_not_stab[w_int[stab_prod_ind]] = False

        # self.product_indexes = np.where(product_c > 0)[0]
        # prod_left_shift = self.product_indexes - 1
        # prod_right_shift = self.product_indexes + 1
        # self.product_indexes = np.unique(np.concatenate((self.product_indexes, prod_left_shift, prod_right_shift)))
        # temp_ind = np.where((self.product_indexes >= 0) & (self.product_indexes <= self.furthest_index))
        # self.product_indexes = self.product_indexes[temp_ind]

        # some = np.where((product_c[self.product_indexes] < self.param["phase_fraction_lim"]) & (product_c[self.product_indexes] > 0))[0]
        # some = np.where(product_c[self.product_indexes] < self.param["phase_fraction_lim"])[0]
        # self.product_indexes = self.product_indexes[some]

        act_ox_mutual_ind = self.get_active_oxidant_mutual_indexes(oxidant, active)
        self.comb_indexes = np.intersect1d(act_ox_mutual_ind, self.nucleation_indexes)

        # oxidant_indexes = np.where(oxidant > 0)[0]
        # active_indexes = np.where(active > 0)[0]
        # min_act = active_indexes.min(initial=self.cells_per_axis)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act - 1)[0]
        #     comb_indexes = oxidant_indexes[indexs]
        #     self.comb_indexes = np.intersect1d(comb_indexes, self.product_indexes)
        # else:
        #     self.comb_indexes = [self.furthest_index]

    def calc_stable_products(self):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_active_moles = secondary_active * Config.ACTIVES.SECONDARY.MOLES_PER_CELL
        secondary_outward_eq_mat_moles = secondary_active * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL_TC
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(self.furthest_index + 1)], dtype=np.uint32)
        secondary_product_moles = secondary_product * Config.PRODUCTS.SECONDARY.MOLES_PER_CELL_TC
        secondary_product_eq_mat_moles = secondary_product * Config.ACTIVES.SECONDARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles -\
                       secondary_outward_eq_mat_moles - secondary_product_eq_mat_moles
        neg_ind = np.where(matrix_moles < 0)[0]
        matrix_moles[neg_ind] = 0
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles + secondary_active_moles +\
                      secondary_product_moles

        product_c = product_moles / whole_moles
        secondary_product_c = secondary_product_moles / whole_moles

        # oxidant_pure = oxidant + product + secondary_product
        # oxidant_pure_moles = oxidant_pure * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        #
        # active_pure = active + product
        # active_pure_moles = active_pure * self.param["active_element"]["primary"]["moles_per_cell"]
        # active_pure_eq_mat_moles = active_pure * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
        #
        # secondary_active_pure = secondary_active + secondary_product
        # secondary_active_pure_moles = secondary_active_pure * self.param["active_element"]["secondary"]["moles_per_cell"]
        # secondary_active_pure_eq_mat_moles = secondary_active_pure * self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"]
        #
        # matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles - secondary_active_pure_eq_mat_moles
        # whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles
        #
        # oxidant_pure_c = oxidant_pure_moles / whole_moles_pure
        # active_pure_c = active_pure_moles / whole_moles_pure
        # secondary_active_pure_c = secondary_active_pure_moles / whole_moles_pure

        oxidant_pure_moles = oxidant_moles + product_moles * 3 + secondary_product_moles * 3

        active_pure_moles = active_moles + product_moles * 2
        active_pure_eq_mat_moles = active_pure_moles * Config.ACTIVES.PRIMARY.T
        secondary_active_pure_moles = secondary_active_moles + secondary_product_moles * 2
        secondary_active_pure_eq_mat_moles = secondary_active_pure_moles * Config.ACTIVES.SECONDARY.T

        matrix_moles_pure = self.matrix_moles_per_page - active_pure_eq_mat_moles - secondary_active_pure_eq_mat_moles
        neg_ind = np.where(matrix_moles_pure < 0)[0]
        matrix_moles_pure[neg_ind] = 0
        whole_moles_pure = matrix_moles_pure + oxidant_pure_moles + active_pure_moles + secondary_active_pure_moles

        oxidant_pure_c = oxidant_pure_moles * 100 / whole_moles_pure
        active_pure_c = active_pure_moles * 100 / whole_moles_pure
        secondary_active_pure_c = secondary_active_pure_moles * 100 / whole_moles_pure

        self.curr_look_up = self.TdDATA.get_look_up_data(active_pure_c, secondary_active_pure_c, oxidant_pure_c)

        primary_diff = self.curr_look_up[0] - product_c
        primary_pos_ind = np.where(primary_diff >= 0)[0]
        primary_neg_ind = np.where(primary_diff < 0)[0]

        secondary_diff = self.curr_look_up[1] - secondary_product_c
        secondary_pos_ind = np.where(secondary_diff >= 0)[0]
        secondary_neg_ind = np.where(secondary_diff < 0)[0]

        self.cur_case = self.cases.first
        if len(primary_pos_ind) > 0:
            oxidant_indexes = np.where(oxidant > 0)[0]
            active_indexes = np.where(active > 0)[0]
            min_act = active_indexes.min(initial=self.cells_per_axis)
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]

            self.comb_indexes = np.intersect1d(primary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cur_case.fix_init_precip_func_ref(self.furthest_index)
                self.precip_step_two_products()
                self.dissolution_zhou_wei_no_bsf()

        if len(primary_neg_ind) > 0:
            self.comb_indexes = primary_neg_ind
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes, np.ones(len(self.comb_indexes)))
            self.dissolution_zhou_wei_no_bsf()
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.zeros(len(self.comb_indexes)))

        self.cur_case = self.cases.second
        if len(secondary_pos_ind) > 0:
            self.get_combi_ind_two_products(self.secondary_active)
            self.comb_indexes = np.intersect1d(secondary_pos_ind, self.comb_indexes)

            if len(self.comb_indexes) > 0:
                self.cur_case.fix_init_precip_func_ref(self.furthest_index)
                self.precip_step_two_products()
                self.dissolution_zhou_wei_no_bsf()

        if len(secondary_neg_ind) > 0:
            self.comb_indexes = secondary_neg_ind
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.ones(len(self.comb_indexes)))
            self.dissolution_zhou_wei_no_bsf()
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes,
                                                                        np.zeros(len(self.comb_indexes)))

    def precipitation_with_td(self):
        self.furthest_index = self.primary_oxidant.calc_furthest_index()

        if self.furthest_index >= self.curr_max_furthest:
            self.curr_max_furthest = self.furthest_index

        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        self.calc_stable_products()
        self.primary_oxidant.transform_to_descards()

    def precipitation_first_case(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            # self.cur_case = self.cases.first
            # self.nucl_prob.adapt_probabilities(self.comb_indexes, self.rel_prod_fraction[self.comb_indexes],
            #                                    self.gamma_primes[self.comb_indexes])
            # self.nucl_prob.adapt_probabilities(self.comb_indexes, self.gamma_primes[self.comb_indexes])
            self.cases.first.fix_init_precip_func_ref(self.furthest_index)
            self.precip_step()

        self.primary_oxidant.transform_to_descards()

    def precipitation_growth_test(self):
        # created to test how growth function ang probabilities work
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.cells_per_axis)

        self.comb_indexes = np.where(self.product_x_nzs)[0]
        prod_left_shift = self.comb_indexes - 1
        prod_right_shift = self.comb_indexes + 1
        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, prod_left_shift, prod_right_shift)))

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        product_conc = product / (self.cells_per_page * self.primary_oxid_numb)

        # middle_ind = np.where(self.comb_indexes == self.mid_point_coord)[0]
        # rel_phase_fraction_for_all = product_conc[middle_ind] / self.param["phase_fraction_lim"]

        some = np.where(product_conc < Config.PHASE_FRACTION_LIMIT)[0]

        self.comb_indexes = self.comb_indexes[some]

        # rel_product_fractions = product_conc[some] / self.param["phase_fraction_lim"]
        # rel_product_fractions[:] = rel_phase_fraction_for_all

        if len(self.comb_indexes) > 0:
            # self.nucl_prob.adapt_probabilities(self.comb_indexes, rel_product_fractions)
            self.cases.first.fix_init_precip_func_ref(self.cells_per_axis)
            self.precip_step()

        self.primary_oxidant.transform_to_descards()

    def precipitation_growth_test_with_p1(self):
        # in this case single probability for growth were given, if at least one product neighbour then nucleation with
        # P1. the probability functions was were adapted accordingly.
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.cells_per_axis)

        self.comb_indexes = np.where(self.product_x_nzs)[0]
        prod_left_shift = self.comb_indexes - 1
        prod_right_shift = self.comb_indexes + 1
        self.comb_indexes = np.unique(np.concatenate((self.comb_indexes, prod_left_shift, prod_right_shift)))

        u_bound = self.comb_indexes.max()
        l_bound = self.comb_indexes.min()

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.comb_indexes], dtype=np.uint32)
        # product_conc = product / (self.cells_per_page * self.primary_oxid_numb)

        # middle_ind = np.where(self.comb_indexes == self.mid_point_coord)[0]
        # rel_phase_fraction_for_all = product_conc[middle_ind] / self.param["phase_fraction_lim"]

        # some = np.where(product_conc < self.param["phase_fraction_lim"])[0]

        # self.comb_indexes = self.comb_indexes[some]

        # rel_product_fractions = product_conc[some] / self.param["phase_fraction_lim"]
        # rel_product_fractions[:] = rel_phase_fraction_for_all

        if len(self.comb_indexes) > 0:
            self.cases.first.fix_init_precip_func_ref(u_bound, l_bound=l_bound)
            self.precip_step()

        self.primary_oxidant.transform_to_descards()

    def precipitation_first_case_no_growth(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind()

        if len(self.comb_indexes) > 0:
            # self.cur_case = self.cases.first
            self.precip_step()
        self.primary_oxidant.transform_to_descards()

    def precipitation_0_cells_no_growth_solub_prod_test(self):
        """
        Created only for tests of the solubility product probability function
        """
        # Only one oxidant and one active elements exist. Only one product can be created
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)

        self.get_combi_ind_atomic_solub_prod_test()

        if len(self.comb_indexes) > 0:
            self.nucl_prob.adapt_probabilities(self.comb_indexes, self.gamma_primes[self.comb_indexes])
            self.precip_step_no_growth_solub_prod_test()
        self.primary_oxidant.transform_to_descards()

    def dissolution_atomic_stop_if_stable(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]
        where_not_stab = np.where(self.product_x_not_stab)[0]
        self.product_indexes = np.intersect1d(self.product_indexes, where_not_stab)

        new_stab_count = np.count_nonzero(~self.product_x_not_stab)
        if new_stab_count > self.prev_stab_count:
            self.prev_stab_count = new_stab_count
            print("stable now at: ", np.nonzero(~self.product_x_not_stab)[0])

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp_ind = np.where(product == 0)[0]
        self.product_x_nzs[self.product_indexes[temp_ind]] = False

        product_c = np.delete(product_c, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        self.product_x_not_stab[self.comb_indexes[temp_ind]] = False

        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_if_stable_higer_p(self):
        self.comb_indexes = np.where(self.product_x_nzs)[0]
        # where_not_stab = np.where(self.product_x_not_stab)[0]
        # self.product_indexes = np.intersect1d(self.product_indexes, where_not_stab)

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.comb_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.comb_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        # temp_ind = np.where(product == 0)[0]
        # self.product_x_nzs[self.product_indexes[temp_ind]] = False

        # product_c = np.delete(product_c, temp_ind)
        # self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(product_c > Config.PHASE_FRACTION_LIMIT)[0]
        # self.product_x_not_stab[self.comb_indexes[temp_ind]] = False
        # self.comb_indexes = np.delete(self.comb_indexes, temp_ind)
        frac = np.zeros(len(self.comb_indexes))
        frac[temp_ind] = 1

        if len(self.comb_indexes) > 0:
            self.cur_case.dissolution_probabilities.adapt_probabilities(self.comb_indexes, frac)
            self.decomposition_intrinsic()

    def dissolution_atomic_stop_if_no_active(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        temp_ind = np.where(product == 0)[0]
        self.product_x_nzs[self.product_indexes[temp_ind]] = False

        active = np.delete(active, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(active == 0)[0]

        # if len(temp_ind) > 0:
        #     print()

        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_stop_if_no_active_or_no_oxidant(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)

        temp_ind = np.where(product == 0)[0]
        self.product_x_nzs[self.product_indexes[temp_ind]] = False

        active = np.delete(active, temp_ind)
        oxidant = np.delete(oxidant, temp_ind)
        self.comb_indexes = np.delete(self.product_indexes, temp_ind)

        temp_ind = np.where(active == 0)[0]
        temp_ind1 = np.where(oxidant == 0)[0]

        temp_ind = np.unique(np.concatenate((temp_ind, temp_ind1)))

        self.comb_indexes = np.delete(self.comb_indexes, temp_ind)

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (product_c > self.soll_prod))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic_and_KP(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.product_indexes]) ** 1.1
        self.diffs = product_c - soll_prod

        # temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (product_c > self.soll_prod))[0]
        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (self.diffs > 0))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()

    def dissolution_atomic_with_kinetic_MP(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        self.primary_oxidant.transform_to_3d()
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL
        self.primary_oxidant.transform_to_descards()

        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in self.product_indexes], dtype=np.uint32)
        active_moles = active * Config.ACTIVES.PRIMARY.MOLES_PER_CELL
        outward_eq_mat_moles = active * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        product_moles = product * Config.PRODUCTS.PRIMARY.MOLES_PER_CELL
        product_eq_mat_moles = product * Config.ACTIVES.PRIMARY.EQ_MATRIX_MOLES_PER_CELL

        matrix_moles = self.matrix_moles_per_page - outward_eq_mat_moles - product_eq_mat_moles
        whole_moles = matrix_moles + oxidant_moles + active_moles + product_moles
        product_c = product_moles / whole_moles

        soll_prod = Config.PROD_INCR_CONST * (self.curr_time - self.active_times[self.product_indexes]) ** 1.1
        self.diffs = product_c - soll_prod

        temp = np.where((product_c > Config.PHASE_FRACTION_LIMIT) | (self.diffs > 0))[0]
        self.comb_indexes = self.product_indexes[temp]

        if len(self.comb_indexes) > 0:
            self.shared_array[:] = self.primary_product.c3d
            tasks = []
            processes = []
            for ran in range(8):
                p = multiprocessing.Process(target=self.decomposition_intrinsic, args=(ran,))
                processes.append(p)
                p.start()

                # tasks.append((range, self.comb_indexes, self.buffer_size * len(self.comb_indexes), self.aggregated_ind))
                # tasks.append((range))

            for p in processes:
                p.join()

            print()
            # results = self.pool.starmap(self.decomposition_intrinsic, tasks)


    def dissolution_test(self):

        self.comb_indexes = np.where(self.product_x_nzs)[0]

        # not_stable_ind = np.where(self.product_x_not_stab)[0]
        # nz_ind = np.where(self.product_x_nzs)[0]

        # self.product_indexes = np.intersect1d(not_stable_ind, nz_ind)

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.product_indexes], dtype=np.uint32)

        # where_no_prod = np.where(product == 0)[0]
        # self.product_x_nzs[self.product_indexes[where_no_prod]] = False

        # self.product_indexes = np.where(self.product_x_nzs)[0]

        if len(self.comb_indexes) > 0:
            self.decomposition_intrinsic()
        # else:
        #     print("PRODUCT FULLY DISSOLVED AFTER ", self.iteration, " ITERATIONS")
        #     sys.exit()

    def precipitation_1(self):
        # ONE oxidant and TWO active elements exist. TWO products can be created.
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        primary_oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        primary_oxidant_mass = primary_oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

        primary_active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                                   in range(furthest_index + 1)], dtype=np.uint32)
        primary_active_mass = primary_active * Config.ACTIVES.PRIMARY.MASS_PER_CELL

        secondary_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
                                     in range(furthest_index + 1)], dtype=np.uint32)
        secondary_active_mass = secondary_active * Config.ACTIVES.SECONDARY.MASS_PER_CELL

        primary_product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        primary_product_mass = primary_product * Config.PRODUCTS.PRIMARY.MASS_PER_CELL

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        secondary_product_mass = secondary_product * Config.PRODUCTS.SECONDARY.MASS_PER_CELL

        matrix_mass = (self.cells_per_axis ** 2 - secondary_active - primary_active - secondary_product -
                       primary_product) * Config.MATRIX.MASS_PER_CELL

        whole = matrix_mass + primary_oxidant_mass + secondary_product_mass + primary_product_mass + \
                secondary_active_mass + primary_active_mass
        # primary_solub_prod = (primary_oxidant_mass * primary_active_mass) / whole ** 2
        # secondary_solub_prod = (primary_oxidant_mass * secondary_active_mass) / whole ** 2

        # if primary_solub_prod >= self.param["sol_prod"]:
        #     self.case = 0
        #     self.precip_step(plane_x_ind)
        #
        # if secondary_solub_prod >= self.param["sol_prod"]:
        #     self.case = 1
        #     self.precip_step(plane_x_ind)

    def precipitation_second_case_cells(self):
        # ONE oxidant and TWO active elements exist. TWO products can be created.
        self.furthest_index = self.primary_oxidant.calc_furthest_index()
        self.primary_oxidant.transform_to_3d()

        if self.iteration % Config.STRIDE == 0:
            if self.furthest_index >= self.curr_max_furthest:
                self.curr_max_furthest = self.furthest_index
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            self.secondary_active.transform_to_3d(self.curr_max_furthest)

        # self.furthest_index = self.primary_oxidant.calc_furthest_index()
        #
        # oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
        #                     in range(self.furthest_index + 1)], dtype=np.uint32)
        # active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
        #                      in range(self.furthest_index + 1)], dtype=np.uint32)
        # s_active = np.array([np.sum(self.secondary_active.c3d[:, :, plane_ind]) for plane_ind
        #                    in range(self.furthest_index + 1)], dtype=np.uint32)
        #
        # oxidant_indexes = np.where(oxidant > 0)[0]
        # active_indexes = np.where(active > 0)[0]
        #
        # min_act = active_indexes.min(initial=self.cells_per_axis)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act - 1)[0]
        #     self.comb_indexes = oxidant_indexes[indexs]
        # else:
        #     self.comb_indexes = [self.furthest_index]
        #
        #
        # np.where(active < )
        self.get_combi_ind_two_products(self.primary_active)

        if len(self.comb_indexes) > 0:
            self.cur_case = self.cases.first
            self.precip_step_two_products()

        self.get_combi_ind_two_products(self.secondary_active)
        if len(self.comb_indexes) > 0:
            self.cur_case = self.cases.second
            self.precip_step_two_products()

        # min_act = primary_active_indexes.min(initial=self.cells_per_axis + 10)
        # if min_act < self.cells_per_axis:
        #     indexs = np.where(oxidant_indexes >= min_act)[0]
        #     self.comb_indexes = oxidant_indexes[indexs]
        # else:
        #     self.comb_indexes = [furthest_index]
        #
        # if len(self.comb_indexes) > 0:
        #     self.cur_case = self.cases.first
        #     self.precip_step_two_products()
        #
        # if len(primary_active_indexes_neg) > 0:
        #     secondary_active_indexes = np.array(np.intersect1d(secondary_active_indexes, primary_active_indexes_neg))
        #     min_act = secondary_active_indexes.min(initial=self.cells_per_axis + 10)
        #     if min_act < self.cells_per_axis:
        #         indexs = np.where(oxidant_indexes >= min_act)[0]
        #         self.comb_indexes = oxidant_indexes[indexs]
        #     else:
        #         self.comb_indexes = [furthest_index]
        #
        #     if len(self.comb_indexes) > 0:
        #         self.cur_case = self.cases.second
        #         self.precip_step_two_products()

        self.primary_oxidant.transform_to_descards()

    def precip_step_standard(self):
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index,
                                                                      dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        # activate if microstructure ___________________________________________________________
                        # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                        # temp_ind = np.where(in_gb)[0]
                        # oxidant_cells = oxidant_cells[temp_ind]
                        # ______________________________________________________________________________________
                        self.check_intersection(oxidant_cells)

    def precip_step_two_products(self):
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    exists = check_at_coord(self.cur_case.to_check_with.c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        # activate if microstructure ___________________________________________________________
                        # in_gb = [self.microstructure[point[0], point[1], point[2]] for point in oxidant_cells]
                        # temp_ind = np.where(in_gb)[0]
                        # oxidant_cells = oxidant_cells[temp_ind]
                        # ______________________________________________________________________________________
                        self.ci_single_two_products(oxidant_cells)

    def precip_step_no_growth(self):
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        self.check_intersection(oxidant_cells)

    def precip_step_no_growth_solub_prod_test(self):
        """
        Created only for tests of the solubility product probability function
        """
        for plane_index in reversed(self.comb_indexes):
            for fetch_ind in self.fetch_ind:
                oxidant_cells = self.cur_case.oxidant.c3d[fetch_ind[0], fetch_ind[1], plane_index]
                oxidant_cells = fetch_ind[:, np.nonzero(oxidant_cells)[0]]

                if len(oxidant_cells[0]) != 0:
                    oxidant_cells = np.vstack((oxidant_cells, np.full(len(oxidant_cells[0]), plane_index, dtype=np.short)))
                    oxidant_cells = oxidant_cells.transpose()

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells)  # precip on place of oxidant!
                    temp_ind = np.where(exists)[0]
                    oxidant_cells = np.delete(oxidant_cells, temp_ind, 0)

                    if len(oxidant_cells) > 0:
                        self.ci_single_no_growth_solub_prod_test(oxidant_cells)

    def ci_single(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ushort)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            flat_arounds = all_arounds[:, 0:self.cur_case.product.lind_flat_arr]
            arr_len_in_flat = self.cur_case.go_around_func_ref(flat_arounds)
            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            needed_prob = self.cases.first.nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob[homogeneous_ind] = self.cases.first.nucleation_probabilities.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
            randomise = np.array(np.random.random_sample(arr_len_in_flat.size), dtype=np.float64)
            temp_ind = np.where(randomise < needed_prob)[0]

            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]

                out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
                to_del = [np.random.choice(item, self.threshold_outward, replace=False) for item in out_to_del]
                coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)

                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                coord = coord.transpose()
                seeds = seeds.transpose()

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                self.product_x_nzs[seeds[2][0]] = True

    def ci_multi(self, seeds):
        """
        Check intersections between the seeds neighbourhood and the coordinates of inward particles only.
        Compute which seed will become a precipitation and which inward particles should be deleted
        according to threshold_inward conditions. This is a simplified version of the check_intersection() function
        where threshold_outward is equal to 1, so there is no need to check intersection with OUT arrays!

        :param seeds: array of seeds coordinates
        """
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = np.array([[self.cur_case.active.c3d[point[0], point[1], point[2]]
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
            exists = [self.cur_case.product.full_c3d[point[0], point[1], point[2]] for point in coord]
            temp_ind = np.where(exists)[0]
            coord = np.delete(coord, temp_ind, 0)
            seeds = np.delete(seeds, temp_ind, 0)

            if self.cur_case.to_check_with is not None:
                exists = [self.cur_case.to_check_with.c3d[point[0], point[1], point[2]] for point in coord]
                temp_ind = np.where(exists)[0]
                coord = np.delete(coord, temp_ind, 0)
                seeds = np.delete(seeds, temp_ind, 0)

            if len(seeds) > 0:
                self_all_arounds = self.utils.calc_sur_ind_formation_noz(seeds, self.cur_case.oxidant.c3d.shape[2] - 1)
                self_neighbours = np.array([[self.cur_case.oxidant.c3d[point[0], point[1], point[2]]
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

                    self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                    self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1
                    self.cur_case.product.fix_full_cells(coord)

                    self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1
                    self.cur_case.oxidant.c3d[coord_in[0], coord_in[1], coord_in[2]] -= 1

    def ci_single_only_p1(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            flat_arounds = all_arounds[:, 0:self.cur_case.product.lind_flat_arr]
            # arr_len_in_flat = self.go_around(self.precipitations3d_init, flat_arounds)
            arr_len_in_flat = self.cur_case.go_around_func_ref(flat_arounds)
            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            needed_prob = np.full(len(arr_len_in_flat), Config.PROBABILITIES.PRIMARY.p1)
            needed_prob[homogeneous_ind] = 0
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

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                self.product_x_nzs[seeds[2][0]] = True

                # self.cumul_product[coord[0], coord[1], coord[2]] += 1

    def ci_single_no_growth_only_p0(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ushort)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            randomise = np.array(np.random.random_sample(len(seeds)), dtype=np.float64)
            temp_ind = np.where(randomise < Config.PROBABILITIES.PRIMARY.p0)[0]
            if len(temp_ind) > 0:
                seeds = seeds[temp_ind]
                neighbours = neighbours[temp_ind]
                all_arounds = all_arounds[temp_ind]
                out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
                to_del = [np.random.choice(item, self.threshold_outward, replace=False) for item in out_to_del]
                coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                                 dtype=np.short)
                coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))
                coord = coord.transpose()
                seeds = seeds.transpose()
                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1
                # self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!
                # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!
                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                self.product_x_nzs[seeds[2][0]] = True

    def ci_single_no_growth(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ushort)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]

            out_to_del = [np.array(np.nonzero(item)[0]) for item in neighbours]
            to_del = [np.random.choice(item, self.threshold_outward, replace=False) for item in out_to_del]
            coord = np.array([all_arounds[seed_ind][point_ind] for seed_ind, point_ind in enumerate(to_del)],
                             dtype=np.short)

            coord = np.reshape(coord, (len(coord) * self.threshold_outward, 3))

            coord = coord.transpose()
            seeds = seeds.transpose()

            self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
            self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

            # self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
            self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

            # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
            # dissolution function
            self.product_x_nzs[seeds[2][0]] = True

    def ci_single_no_growth_solub_prod_test(self, seeds):
        """
        Created only for tests of the solubility product probability function
        """
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            # flat_arounds = all_arounds[:, 0:self.objs[self.case]["product"].lind_flat_arr]

            # flat_neighbours = self.go_around(self.precipitations3d_init_full, flat_arounds)
            # flat_neighbours = self.go_around(flat_arounds)
            # arr_len_in_flat = np.array([np.sum(item) for item in flat_neighbours], dtype=int)

            # arr_len_in_flat = self.go_around(self.precipitations3d_init, flat_arounds)

            # arr_len_in_flat = np.zeros(len(flat_arounds))  # REMOVE!!!!!!!!!!!!!!!!!!

            # homogeneous_ind = np.where(arr_len_in_flat == 0)[0]

            # needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob = self.nucl_prob.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
            randomise = np.array(np.random.random_sample(len(seeds)), dtype=np.float64)
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

                coord = coord.transpose()
                seeds = seeds.transpose()

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

    def ci_single_two_products(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

        # activate for dependent growth___________________________________________________________________
        if len(temp_ind) > 0:
            seeds = seeds[temp_ind]
            neighbours = neighbours[temp_ind]
            all_arounds = all_arounds[temp_ind]
            flat_arounds = all_arounds[:, 0:self.cur_case.product.lind_flat_arr]
            arr_len_in_flat = self.cur_case.go_around_func_ref(flat_arounds)
            homogeneous_ind = np.where(arr_len_in_flat == 0)[0]
            needed_prob = self.cur_case.nucleation_probabilities.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob[homogeneous_ind] = self.cur_case.nucleation_probabilities.nucl_prob.values_pp[
                seeds[0][2]]  # seeds[0][2] - current plane index
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

                self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
                self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

                # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
                self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

                # self.objs[self.case]["product"].fix_full_cells(coord)  # precip on place of active!
                self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

                # mark the x-plane where the precipitate has happened, so the index of this plane can be called in the
                # dissolution function
                self.cur_case.prod_indexes[seeds[2][0]] = True

    def ci_single_two_products_no_growth(self, seeds):
        all_arounds = self.utils.calc_sur_ind_formation(seeds, self.cur_case.active.c3d.shape[2] - 1)
        neighbours = go_around_bool(self.cur_case.active.c3d, all_arounds)
        arr_len_out = np.array([np.sum(item) for item in neighbours], dtype=np.ubyte)
        temp_ind = np.where(arr_len_out >= self.threshold_outward)[0]

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

            self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
            self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

            # self.objs[self.case]["product"].c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
            self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

    def diffusion_inward(self):
        self.primary_oxidant.diffuse()
        if Config.OXIDANTS.SECONDARY_EXISTENCE:
            self.secondary_oxidant.diffuse()

    def diffusion_outward(self):
        if (self.iteration + 1) % Config.STRIDE == 0:
            self.primary_active.transform_to_descards()
            self.primary_active.diffuse()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.secondary_active.transform_to_descards()
                self.secondary_active.diffuse()

    def diffusion_outward_with_mult_srtide(self):
        if self.iteration % Config.STRIDE == 0:
            if self.iteration % self.precipitation_stride == 0 or self.iteration == 0:
                self.primary_active.transform_to_descards()
            self.primary_active.diffuse()

            # if Config.ACTIVES.SECONDARY_EXISTENCE:
            #     self.secondary_active.transform_to_descards()
            #     self.secondary_active.diffuse()

    def calc_precip_front_1(self, iteration):
        """
        Calculating a position of a precipitation front. As a boundary a precipitation concentration of 0,1% is used.
        :param iteration: current iteration (serves as a current simulation time)
        """
        oxidant = self.primary_oxidant.cells
        oxidant = np.array([len(np.where(oxidant[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        oxidant_mass = oxidant * Config.OXIDANTS.PRIMARY.MASS_PER_CELL

        active = self.primary_active.cells
        active = np.array([len(np.where(active[2] == i)[0]) for i in range(self.cells_per_axis)], dtype=int)
        active_mass = active * Config.ACTIVES.PRIMARY.MASS_PER_CELL

        secondary_active = self.secondary_active.cells
        secondary_active = np.array([len(np.where(secondary_active[2] == i)[0]) for i in range(self.cells_per_axis)],
                                    dtype=int)
        secondary_active_mass = secondary_active * Config.ACTIVES.SECONDARY.MASS_PER_CELL

        primary_product = np.array(np.nonzero(self.primary_product.c3d), dtype=int)
        primary_product = np.array([len(np.where(primary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                   dtype=int)
        primary_product_mass = primary_product * Config.PRODUCTS.PRIMARY.MASS_PER_CELL

        secondary_product = np.array(np.nonzero(self.secondary_product.c3d), dtype=int)
        secondary_product = np.array([len(np.where(secondary_product[2] == i)[0]) for i in range(self.cells_per_axis)],
                                     dtype=int)
        secondary_product_mass = secondary_product * Config.PRODUCTS.SECONDARY.MASS_PER_CELL

        matrix_mass = (self.cells_per_axis ** 2 - active - secondary_active - primary_product - secondary_product) * \
                      Config.MATRIX.MASS_PER_CELL
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

    def calc_precip_front_2_cells(self):
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.cells_per_axis)], dtype=np.uint32)

        secondary_product = np.array([np.sum(self.secondary_product.c3d[:, :, plane_ind]) for plane_ind
                                      in range(self.cells_per_axis)], dtype=np.uint32)
        product = product / (self.cells_per_axis ** 2)
        secondary_product = secondary_product / (self.cells_per_axis ** 2)
        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        secondary_threshold = Config.ACTIVES.SECONDARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 2:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 / self.cells_per_axis
                sqr_time = ((self.iteration + 1) * Config.SIM_TIME / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                self.utils.db.conn.commit()
                break
        for rev_index, precip_conc in enumerate(np.flip(secondary_product)):
            if precip_conc > secondary_threshold / 2:
                position = (len(secondary_product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((self.iteration + 1) * Config.SIM_TIME / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "s")
                self.utils.db.conn.commit()
                break

    def calc_precipitation_front_only_cells(self):
        """
        Calculating a position of a precipitation front, considering only cells concentrations without any scaling!
        As a boundary a product fraction of 0,1% is used.
        """
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.cells_per_axis)], dtype=np.uint32)
        product = product / (self.cells_per_axis ** 2)
        threshold = Config.ACTIVES.PRIMARY.CELLS_CONCENTRATION
        for rev_index, precip_conc in enumerate(np.flip(product)):
            if precip_conc > threshold / 100:
                position = (len(product) - 1 - rev_index) * Config.SIZE * 10 ** 6 \
                           / self.cells_per_axis
                sqr_time = ((self.iteration + 1) * Config.SIM_TIME / (self.n_iter * 3600)) ** (1 / 2)
                self.utils.db.insert_precipitation_front(sqr_time, position, "p")
                break

    def save_results(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.primary_active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.secondary_active.transform_to_descards()
        if Config.INWARD_DIFFUSION:
            self.utils.db.insert_particle_data("primary_oxidant", self.iteration, self.primary_oxidant.cells)
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.utils.db.insert_particle_data("secondary_oxidant", self.iteration, self.secondary_oxidant.cells)
        if Config.OUTWARD_DIFFUSION:
            self.utils.db.insert_particle_data("primary_active", self.iteration, self.primary_active.cells)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.utils.db.insert_particle_data("secondary_active", self.iteration, self.secondary_active.cells)
        if Config.COMPUTE_PRECIPITATION:
            self.utils.db.insert_particle_data("primary_product", self.iteration, self.primary_product.transform_c3d())
            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.utils.db.insert_particle_data("secondary_product", self.iteration,
                                                   self.secondary_product.transform_c3d())
                self.utils.db.insert_particle_data("ternary_product", self.iteration,
                                                   self.ternary_product.transform_c3d())
                self.utils.db.insert_particle_data("quaternary_product", self.iteration,
                                                   self.quaternary_product.transform_c3d())
            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.utils.db.insert_particle_data("secondary_product", self.iteration, self.secondary_product.transform_c3d())
        if Config.STRIDE > Config.N_ITERATIONS:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
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

    def fix_init_precip_bool(self, u_bound, l_bound=0):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        if l_bound - 1 < 0:
            l_bound = 1
        self.cur_case.precip_3d_init[:, :, l_bound-1:u_bound + 2] = False
        self.cur_case.precip_3d_init[:, :, l_bound-1:u_bound + 2] = self.cur_case.product.c3d[:, :, l_bound-1:u_bound + 2]

    def fix_init_precip_int(self, u_bound):
        if u_bound == self.cells_per_axis - 1:
            u_bound = self.cells_per_axis - 2
        self.cur_case.precip_3d_init[:, :, 0:u_bound + 2] = 0
        self.cur_case.precip_3d_init[:, :, 0:u_bound + 2] = self.cur_case.product.c3d[:, :, 0:u_bound + 2]

    def fix_init_precip_dummy(self, u_bound, l_bound=0):
        pass

    def get_active_oxidant_mutual_indexes(self, oxidant, active):
        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            index = np.where(oxidant_indexes >= min_act - 1)[0]
            return oxidant_indexes[index]
        else:
            return [self.furthest_index]

    def go_around_single_oxid_n(self, around_coords):
        return np.sum(go_around_bool(self.cur_case.precip_3d_init, around_coords), axis=1)

    def go_around_mult_oxid_n(self, around_coords):
        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        where_full_side_neigh = np.unique(np.where(all_neigh[:, :6].view() == self.cur_case.product.oxidation_number)[0])
        only_inside_product = np.setdiff1d(nonzero_neigh_in_prod, where_full_side_neigh, assume_unique=True)
        final_effective_flat_counts = np.zeros(len(all_neigh), dtype=np.ubyte)
        final_effective_flat_counts[where_full_side_neigh] = np.sum(all_neigh[where_full_side_neigh], axis=1)
        final_effective_flat_counts[only_inside_product] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts

    def go_around_mult_oxid_n_also_partial_neigh(self, around_coords):
        """Im Gegensatz zu go_around_mult_oxid_n werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!
        Resolution inside a product: If inside a product the probability is equal to ONE!!"""
        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        final_effective_flat_counts = np.sum(all_neigh, axis=1)
        final_effective_flat_counts[nonzero_neigh_in_prod] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts

    def go_around_mult_oxid_n_also_partial_neigh_aip(self, around_coords):
        """Im Gegensatz zu go_around_mult_oxid_n werden auch die parziellen Nachbarn (weniger als oxidation numb inside)
        berücksichtigt!!!
        aip: Adjusted Inside Product!
        Resolution inside a product: probability adjusted according to a number of neighbours"""
        return np.sum(go_around_int(self.cur_case.precip_3d_init, around_coords), axis=1)

    def go_around_single_oxid_n_single_neigh(self, around_coords):
        """Does not distinguish between multiple flat neighbours. If at least one flat neighbour P=P1"""
        flat_neighbours = go_around_bool(self.cur_case.precip_3d_init, around_coords)
        temp = np.array([np.sum(item) for item in flat_neighbours], dtype=bool)

        return np.array(temp, dtype=np.ubyte)

    def go_around_mult_oxid_n_single_neigh(self, around_coords):
        """Does not distinguish between multiple flat neighbours. If at least one flat neighbour P=P1"""

        all_neigh = go_around_int(self.cur_case.precip_3d_init, around_coords)
        neigh_in_prod = all_neigh[:, 6].view()
        nonzero_neigh_in_prod = np.array(np.nonzero(neigh_in_prod)[0])
        where_full_side_neigh = np.unique(np.where(all_neigh[:, :6].view() == self.cur_case.product.oxidation_number)[0])
        only_inside_product = np.setdiff1d(nonzero_neigh_in_prod, where_full_side_neigh, assume_unique=True)
        final_effective_flat_counts = np.zeros(len(all_neigh), dtype=np.ubyte)
        final_effective_flat_counts[where_full_side_neigh] = self.cur_case.product.oxidation_number
        final_effective_flat_counts[only_inside_product] = 7 * self.cur_case.product.oxidation_number - 1
        return final_effective_flat_counts

    def generate_fetch_ind(self):
        size = 3 + (Config.NEIGH_RANGE - 1) * 2
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

    def insert_last_it(self):
        self.utils.db.insert_last_iteration(self.iteration)
