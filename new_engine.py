from utils.my_data_structs import *
from utils.new_utils import *
import progressbar
from new_elements import *
import time
from utils.templates import *
from utils.new_probabilities import *


class CellularAutomata:
    def __init__(self):
        self.utils = Utils()
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

        self.cases = CaseRef()
        self.cur_case = None

        # setting objects for inward diffusion
        if Config.INWARD_DIFFUSION:
            self.primary_oxidant = OxidantElem(Config.OXIDANTS.PRIMARY, self.utils)
            self.cases.first.oxidant = self.primary_oxidant
            self.cases.second.oxidant = self.primary_oxidant
            # ---------------------------------------------------
            # self.primary_oxidant.diffuse = None  # must be defined elsewhere
            # self.primary_oxidant.diffuse = self.primary_oxidant.diffuse_with_scale
            # self.primary_oxidant.diffuse = self.primary_oxidant.diffuse_bulk
            # self.primary_oxidant.diffuse = self.primary_oxidant.diffuse_gb
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
            # self.primary_active.diffuse = None  # must be defined elsewhere
            # self.primary_active.diffuse = self.primary_active.diffuse_with_scale  # must be defined elsewhere
            # self.primary_active.diffuse = self.primary_active.diffuse_bulk
            # ---------------------------------------------------
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.secondary_active = ActiveElem(Config.ACTIVES.SECONDARY)
                self.cases.second.active = self.secondary_active
                self.cases.fourth.active = self.secondary_active
                # ---------------------------------------------------
                # self.secondary_active.diffuse = self.secondary_active.diffuse_with_scale
                # self.secondary_active.diffuse = self.secondary_active.diffuse_bulk
                # ---------------------------------------------------
        # setting objects for precipitations
        if Config.COMPUTE_PRECIPITATION:
            self.precip_func = None  # must be defined elsewhere
            self.get_combi_ind = None  # must be defined elsewhere
            self.precip_step = None
            self.check_intersection = None  # must be defined elsewhere
            self.decomposition = None  # must be defined elsewhere

            self.coord_buffer = MyBufferCoords(Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER * self.cells_per_axis ** 3)
            self.to_dissol_pn_buffer = MyBufferCoords(Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER * self.cells_per_axis ** 3)

            self.primary_product = Product(Config.PRODUCTS.PRIMARY)
            self.cases.first.product = self.primary_product

            self.primary_oxid_numb = Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER
            self.max_inside_neigh_number = 6 * self.primary_oxid_numb
            self.max_block_neigh_number = 7 * self.primary_oxid_numb

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

                # self.precip_func = self.precipitation_second_case_cells
                # self.calc_precip_front = self.calc_precip_front_1
                # self.decomposition = self.decomposition_0

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
                # self.precip_func = self.precipitation_first_case_no_neigbouring_area  # CHANGE!!!!!!!!!!!!
                # self.precip_func = self.precipitation_first_case
                # self.precip_func = self.precipitation_first_case_no_growth
                # self.precip_func = self.precipitation_0
                # self.calc_precip_front = self.calc_precip_front_0
                # self.decomposition = self.dissolution_0_cells
                self.primary_oxidant.scale = self.primary_product
                self.primary_active.scale = self.primary_product

                if self.cases.first.product.oxidation_number == 1:
                    self.cases.first.go_around_func_ref = self.go_around_single_oxid_n
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_bool
                    self.cases.first.precip_3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1), False, dtype=bool)
                else:
                    self.cases.first.go_around_func_ref = self.go_around_mult_oxid_n
                    self.cases.first.fix_init_precip_func_ref = self.fix_init_precip_int
                    self.cases.first.precip_3d_init = np.full(
                        (self.cells_per_axis, self.cells_per_axis, self.cells_per_axis + 1), 0, dtype=np.ubyte)
            # self.precipitations3d = np.full(self.shape, False)
            # self.half_thickness = 20
            # middle = int(self.cells_per_axis / 2)
            # minus = middle - self.half_thickness
            # plus = middle + self.half_thickness
            # self.primary_product.c3d[minus:plus, minus:plus, minus:plus] = 4
            #
            # self.primary_product.c3d[1, 1, 50] = 4
            # self.primary_product.full_c3d[1, 1, 50] = True
            #
            # self.primary_product.c3d[1, 1, 49] = 4
            # self.primary_product.full_c3d[1, 1, 49] = True
            #
            # self.primary_product.c3d[1, 2, 49] = 4
            # self.primary_product.full_c3d[1, 2, 49] = True
            #
            # self.primary_product.c3d[1, 2, 50] = 4
            # self.primary_product.full_c3d[1, 2, 50] = True
            #
            #
            # self.primary_product.c3d[2, 1, 50] = 4
            # self.primary_product.full_c3d[2, 1, 50] = True
            #
            # self.primary_product.c3d[2, 1, 49] = 4
            # self.primary_product.full_c3d[2, 1, 49] = True
            #
            # self.primary_product.c3d[2, 2, 49] = 4
            # self.primary_product.full_c3d[2, 2, 49] = True
            #
            # self.primary_product.c3d[2, 2, 50] = 4
            # self.primary_product.full_c3d[2, 2, 50] = True
            #
            #
            # self.primary_product.c3d[2, 0, 50] = 4
            # self.primary_product.full_c3d[2, 0, 50] = True
            #
            # self.primary_product.c3d[2, 1, 51] = 4
            # self.primary_product.full_c3d[2, 1, 51] = True
            #
            # self.primary_product.c3d[3, 1, 50] = 4
            # self.primary_product.full_c3d[3, 1, 50] = True
            #
            #
            # self.primary_product.c3d[3, 3, 50] = 2
            # self.primary_product.c3d[10, 3, 50] = 2
            # self.primary_product.c3d[35, 3, 50] = 2
            #
            # self.primary_product.c3d[45, 3, 50] = 4
            # self.primary_product.full_c3d[45, 3, 50] = True
            #
            # self.primary_product.full_c3d[minus:plus, minus:plus, minus:plus] = True
            # self.primary_product.full_c3d[1, 1, 50] = True
            # shift = 0
            # self.precipitations3d[minus + shift:plus + shift, minus + shift:plus + shift, minus + shift:plus + shift] = True
            # self.precipitations = np.array(np.nonzero(self.precipitations), dtype=int)
            # self.precipitations3d = np.full(self.single_page_shape, False)
            # self.precipitations3d_sec = np.full(self.single_page_shape, False)
            self.threshold_inward = Config.THRESHOLD_INWARD
            self.threshold_outward = Config.THRESHOLD_OUTWARD

            self.fetch_ind = None
            self.generate_fetch_ind()

            self.aggregated_ind = [[7, 0, 1, 2, 19, 16, 14],
                                   [6, 0, 1, 5, 18, 15, 14],
                                   [8, 0, 4, 5, 20, 15, 17],
                                   [9, 0, 4, 2, 21, 16, 17],
                                   [11, 3, 1, 2, 19, 24, 22],
                                   [10, 3, 1, 5, 18, 23, 22],
                                   [12, 3, 4, 5, 20, 23, 25],
                                   [13, 3, 4, 2, 21, 24, 25]]

            # UNCOMENT!!!!_______________________________________________________
            self.nucl_prob = NucleationProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.PRIMARY)
            self.dissol_prob = DissolutionProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.PRIMARY)
            # ________________________________________________________________________

            self.furthest_index = 0
            self.comb_indexes = None
            self.rel_prod_fraction = None
            self.gamma_primes = None
            self.product_indexes = None
            self.nucleation_indexes = None

            self.save_flag = False
            self.product_x_nzs = np.full(self.cells_per_axis, False, dtype=bool)
            self.product_x_not_stab = np.full(self.cells_per_axis, True, dtype=bool)
            # self.cumul_prod = np.empty(self.n_iter, dtype=int)
            # self.mid_point_coord = int((Config.N_CELLS_PER_AXIS - 1) / 2)
            # self.look_up = td_data.TDATA()
            # self.look_up.gen_table_dict()

        self.begin = time.time()

    def simulation(self):
        for self.iteration in progressbar.progressbar(range(self.n_iter)):
            self.precip_func()
            self.decomposition()
            self.diffusion_inward()
            self.diffusion_outward()
            # self.decomposition()
            # self.calc_precipitation_front_only_cells()

            # print()
            # print("left: ", np.sum(self.primary_product.full_c3d[:, :, :44]))
            # print("right: ", np.sum(self.primary_product.full_c3d[:, :, 44:]))
            # print()

            # self.cumul_prod[self.iteration] = np.sum(self.primary_product.c3d)

            # self.save_results()
            # self.utils.db.conn.commit()

            # if self.param["compute_precipitations"]:
            #     self.precip_func()
            #     # self.dissolution_0_cells()
            # # if self.param["decompose_precip"]:
            # #     self.decomposition()
            # if self.param["inward_diffusion"]:
            #     self.diffusion_inward()
            # if self.param["outward_diffusion"]:
            #     self.diffusion_outward()
            #
            # if self.param["save_whole"]:
            #     self.save_results_only_prod()
            #
            # if self.iteration % 100 == 0:
            #     self.calc_precip_front_2_cells()

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
        # plane_indxs = np.array([50])
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

    def dissolution_zhou_wei_no_bsf(self):
        """
        Implementation of adjusted Zhou and Wei approach. Only side neighbours are checked. No need for block scale
        factor. Works only for any oxidation nuber!
        """
        nz_ind = np.array(np.nonzero(self.primary_product.c3d[:, :, self.product_indexes]))
        self.coord_buffer.copy_to_buffer(nz_ind)
        self.coord_buffer.update_buffer_at_axis(self.product_indexes[nz_ind[2]], axis=2)

        if self.coord_buffer.last_in_buffer > 0:
            all_arounds = self.utils.calc_sur_ind_decompose_flat(self.coord_buffer.get_buffer())
            all_neigh = go_around_int(self.primary_product.c3d, all_arounds)

            all_neigh_pn = all_neigh[[]]
            # all_neigh_block = all_neigh[[]]

            # choose all the coordinates which have at least one full side neighbour
            where_full = np.unique(np.where(all_neigh == self.primary_oxid_numb)[0])

            to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))
            # self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(where_full))

            if self.coord_buffer.last_in_buffer > 0:
                all_neigh = all_neigh[where_full]

                arr_len_flat = np.array([np.sum(item) for item in all_neigh], dtype=np.ubyte)
                index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]

                self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                all_neigh_pn = arr_len_flat[index_outside]
            else:
                all_neigh_pn = np.array([np.sum(item) for item in all_neigh_pn])

                # self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
                # all_neigh = all_neigh[index_outside]

                # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh],
                #                        dtype=np.ubyte)
                # ind_where_blocks = np.unique(np.where(aggregation == self.max_block_neigh_number)[0])

                # if len(ind_where_blocks) > 0:
                #     self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
                #     all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)

                    # self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
                    # all_neigh_block = all_neigh[ind_where_blocks]
                # else:
                    # self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
                    # all_neigh_pn = all_neigh

                    # self.coord_buffer.reset_buffer()
                    # all_neigh_block = all_neigh[[]]

            probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

            to_dissolve_pn = self.to_dissol_pn_buffer.get_buffer()
            probs_pn = self.dissol_prob.get_probabilities(all_neigh_pn, to_dissolve_pn[2])

            # to_dissolve_p = self.coord_buffer.get_buffer()
            # all_neigh_block = np.array([np.sum(item[:6]) for item in all_neigh_block])
            # probs_p = self.dissol_prob.get_probabilities(all_neigh_block, to_dissolve_p[2])

            randomise = np.random.random_sample(len(to_dissol_pn_no_neigh[0]))
            temp_ind = np.where(randomise < probs_pn_no_neigh)[0]
            to_dissol_pn_no_neigh = to_dissol_pn_no_neigh[:, temp_ind]

            randomise = np.random.random_sample(len(to_dissolve_pn[0]))
            temp_ind = np.where(randomise < probs_pn)[0]
            to_dissolve_pn = to_dissolve_pn[:, temp_ind]

            # randomise = np.random.random_sample(len(to_dissolve_p[0]))
            # temp_ind = np.where(randomise < probs_p)[0]
            # to_dissolve_p = to_dissolve_p[:, temp_ind]

            # to_dissolve = np.concatenate((to_dissolve_pn, to_dissolve_p, to_dissol_pn_no_neigh), axis=1)
            to_dissolve = np.concatenate((to_dissolve_pn, to_dissol_pn_no_neigh), axis=1)

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
            # all_arounds = self.utils.calc_sur_ind_decompose_flat(self.coord_buffer.get_buffer())
            # all_neigh = go_around_int(self.primary_product.c3d, all_arounds)
            #
            # all_neigh_pn = all_neigh[[]]
            # # all_neigh_block = all_neigh[[]]
            #
            # # choose all the coordinates which have at least one full side neighbour
            # where_full = np.unique(np.where(all_neigh == self.primary_oxid_numb)[0])
            #
            # to_dissol_pn_no_neigh = np.array(self.coord_buffer.get_elem_instead_ind(where_full), dtype=np.short)
            # self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(where_full))
            # # self.to_dissol_pn_buffer.append_to_buffer(self.coord_buffer.get_elem_instead_ind(where_full))
            #
            # if self.coord_buffer.last_in_buffer > 0:
            #     all_neigh = all_neigh[where_full]
            #
            #     arr_len_flat = np.array([np.sum(item) for item in all_neigh], dtype=np.ubyte)
            #     index_outside = np.where((arr_len_flat < self.max_inside_neigh_number))[0]
            #
            #     self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
            #     all_neigh_pn = arr_len_flat[index_outside]
            # else:
            #     all_neigh_pn = np.array([np.sum(item) for item in all_neigh_pn])
            #
            #     # self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(index_outside))
            #     # all_neigh = all_neigh[index_outside]
            #
            #     # aggregation = np.array([[np.sum(item[step]) for step in self.aggregated_ind] for item in all_neigh],
            #     #                        dtype=np.ubyte)
            #     # ind_where_blocks = np.unique(np.where(aggregation == self.max_block_neigh_number)[0])
            #
            #     # if len(ind_where_blocks) > 0:
            #     #     self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_elem_instead_ind(ind_where_blocks))
            #     #     all_neigh_pn = np.delete(all_neigh, ind_where_blocks, axis=0)
            #
            #         # self.coord_buffer.copy_to_buffer(self.coord_buffer.get_elem_at_ind(ind_where_blocks))
            #         # all_neigh_block = all_neigh[ind_where_blocks]
            #     # else:
            #         # self.to_dissol_pn_buffer.copy_to_buffer(self.coord_buffer.get_buffer())
            #         # all_neigh_pn = all_neigh
            #
            #         # self.coord_buffer.reset_buffer()
            #         # all_neigh_block = all_neigh[[]]

            # probs_pn_no_neigh = self.dissol_prob.dissol_prob.values_pp[to_dissol_pn_no_neigh[2]]

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

    def precipitation_0(self):
        # Only one oxidant and one active elements exist. Only one product can be created
        furthest_index = self.primary_oxidant.calc_furthest_index()
        if furthest_index > self.curr_max_furthest:
            self.curr_max_furthest = furthest_index
        self.primary_oxidant.transform_to_3d()
        if self.iteration % Config.STRIDE == 0:
            self.primary_active.transform_to_3d(self.curr_max_furthest)
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        oxidant_mass = oxidant * Config.OXIDANTS.PRIMARY.MASS_PER_CELL
        active = np.array([np.sum(self.primary_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(furthest_index + 1)], dtype=np.uint32)
        active_mass = active * Config.ACTIVES.PRIMARY.MASS_PER_CELL
        product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
                            in range(furthest_index + 1)], dtype=np.uint32)
        product_mass = product * Config.PRODUCTS.PRIMARY.MASS_PER_CELL
        pure_matrix = self.cells_per_page * Config.PRODUCTS.PRIMARY.OXIDATION_NUMBER - active - product
        less_than_zero = np.where(pure_matrix < 0)[0]
        pure_matrix[less_than_zero] = 0
        matrix_mass = pure_matrix * Config.ACTIVES.PRIMARY.EQ_MATRIX_MASS_PER_CELL

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

        plane_indexes = np.array(np.where(solub_prod >= Config.SOL_PROD)[0])

        if len(plane_indexes) > 0:
            # self.fix_init_precip(furthest_index, self.primary_product, self.primary_oxidant.cut_shape)
            self.precip_step()
        self.primary_oxidant.transform_to_descards()

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

        # oxidant_c = oxidant_moles / whole_moles
        # active_c = active_moles / whole_moles
        product_c = product_moles / whole_moles

        # self.rel_prod_fraction = product_c / self.param["phase_fraction_lim"]

        # self.product_indexes = np.where((product_c < self.param["phase_fraction_lim"]) & (product_c > 0))[0]
        self.product_indexes = np.where(product_c > 0)[0]
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

    def get_combi_ind_two_products(self, current_active):
        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in range(self.furthest_index + 1)], dtype=np.uint32)
        active = np.array([np.sum(current_active.c3d[:, :, plane_ind]) for plane_ind
                           in range(self.furthest_index + 1)], dtype=np.uint32)

        oxidant_indexes = np.where(oxidant > 0)[0]
        active_indexes = np.where(active > 0)[0]

        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            indexs = np.where(oxidant_indexes >= min_act - 1)[0]
            self.comb_indexes = oxidant_indexes[indexs]
        else:
            self.comb_indexes = [self.furthest_index]

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
        # print(w_int)

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

        # oxidant_c = oxidant_moles / whole_moles
        # active_c = active_moles / whole_moles
        product_c = product_moles / whole_moles

        # self.product_indexes = np.where((product_c < self.param["phase_fraction_lim"]) & (product_c > 0))[0]
        self.nucleation_indexes = w_int[np.where(product_c < Config.PHASE_FRACTION_LIMIT)[0]]

        stab_prod_ind = np.where(product_c >= Config.PHASE_FRACTION_LIMIT)[0]
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
            self.cur_case = self.cases.first
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
            self.cur_case = self.cases.first
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

    def dissolution_0_cells(self):
        self.product_indexes = np.where(self.product_x_nzs)[0]

        oxidant = np.array([np.sum(self.primary_oxidant.c3d[:, :, plane_ind]) for plane_ind
                            in self.product_indexes], dtype=np.uint32)
        oxidant_moles = oxidant * Config.OXIDANTS.PRIMARY.MOLES_PER_CELL

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

        # oxidant_c = oxidant_moles / whole_moles
        # active_c = active_moles / whole_moles
        product_c = product_moles / whole_moles

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind])
        #                     for plane_ind in nz_prod_plane], dtype=np.uint32)

        # prod_fraction = product_c / self.cells_per_page

        # delt_prod_fraction = prod_fraction - self.prev_prod_fraction[nz_prod_plane]

        # self.cumul_prod_fraction[nz_prod_plane] += delt_prod_fraction

        rel_prod_fraction = product_c / Config.PHASE_FRACTION_LIMIT
        tem_ind = np.where(rel_prod_fraction >= 0.25)[0]

        rel_prod_fraction[tem_ind] = 1
        if len(tem_ind) > 0 and not self.save_flag:
            self.save_flag = True
            print("dissolution began at iter: ", self.iteration)
        # adapt_indexes = self.product_indexes[tem_ind]

        # self.ripening_not_started[self.product_indexes[tem_ind]] = False

        # temp_ind = np.where(self.ripening_not_started)[0]
        # temp_ind = np.where(self.product_indexes == adapt_indexes)[0]
        # self.rel_prod_fraction = self.rel_prod_fraction[temp_ind]

        # temp_ind = np.where(self.cumul_prod_fraction[nz_prod_plane] >
        #                     self.prod_increment_const * self.tau * (self.iteration + 1))[0]

        # soll_prod = self.prod_increment_const * (self.tau * (self.iteration + 1))**2
        # error = prod_fraction / soll_prod

        # temp_ind1 = np.where(error > self.error_prod_conc)[0]

        # temp_ind2 = np.where(prod_fraction > self.param["phase_fraction_lim"])[0]

        # temp_ind = np.unique(np.concatenate((temp_ind1, temp_ind2)))

        # if len(temp_ind) > 0:
        #     print()

        # self.product_indexes = nz_prod_plane[temp_ind]
        # self.product_indexes = np.where(self.product_x_nzs)[0]

        # self.cumul_prod_fraction[nz_prod_plane[temp_ind]] = 0

        self.dissol_prob.adapt_probabilities(self.product_indexes, rel_prod_fraction)
        self.dissolution_zhou_wei_no_bsf()

        # self.cumul_prod[self.iteration] = np.sum(self.primary_product.c3d[:, :, 0])/self.cells_per_page
        # self.growth_rate[self.iteration] = self.prod_increment_const * (self.tau * (self.iteration + 1))**2

    def dissolution_test(self):
        not_stable_ind = np.where(self.product_x_not_stab)[0]
        nz_ind = np.where(self.product_x_nzs)[0]

        self.product_indexes = np.intersect1d(not_stable_ind, nz_ind)

        # product = np.array([np.sum(self.primary_product.c3d[:, :, plane_ind]) for plane_ind
        #                     in self.product_indexes], dtype=np.uint32)

        # where_no_prod = np.where(product == 0)[0]
        # self.product_x_nzs[self.product_indexes[where_no_prod]] = False

        # self.product_indexes = np.where(self.product_x_nzs)[0]

        if len(self.product_indexes) > 0:
            self.dissolution_zhou_wei_no_bsf()
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

                    exists = check_at_coord(self.cur_case.product.full_c3d, oxidant_cells) # precip on place of oxidant!
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
                        self.ci_single_two_products_no_growth(oxidant_cells)

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
            needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob[homogeneous_ind] = self.nucl_prob.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
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

    def ci_single_no_growth(self, seeds):
        """
        Created only for tests of saturation of the IOZ when permeability of active element is sufficient
        :param seeds:
        :return:
        """
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

            coord = coord.transpose()
            seeds = seeds.transpose()

            self.cur_case.active.c3d[coord[0], coord[1], coord[2]] -= 1
            self.cur_case.oxidant.c3d[seeds[0], seeds[1], seeds[2]] -= 1

            # self.cur_case.product.c3d[coord[0], coord[1], coord[2]] += 1  # precip on place of active!
            self.cur_case.product.c3d[seeds[0], seeds[1], seeds[2]] += 1  # precip on place of oxidant!

            # self.cur_case.product.fix_full_cells(coord)  # precip on place of active!
            self.cur_case.product.fix_full_cells(seeds)  # precip on place of oxidant!

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
            needed_prob = self.nucl_prob.get_probabilities(arr_len_in_flat, seeds[0][2])
            needed_prob[homogeneous_ind] = self.nucl_prob.nucl_prob.values_pp[seeds[0][2]]  # seeds[0][2] - current plane index
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
            if precip_conc > threshold / 1000:
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

    def get_active_oxidant_mutual_indexes(self, oxidant_ind, active_ind):
        oxidant_indexes = np.where(oxidant_ind > 0)[0]
        active_indexes = np.where(active_ind > 0)[0]
        min_act = active_indexes.min(initial=self.cells_per_axis)
        if min_act < self.cells_per_axis:
            index = np.where(oxidant_indexes >= min_act - 1)[0]
            return oxidant_indexes[index]
        else:
            return [self.furthest_index]

    def go_around_single_oxid_n(self, around_coords):
        flat_neighbours = go_around_bool(self.cur_case.precip_3d_init, around_coords)
        return np.array([np.sum(item) for item in flat_neighbours], dtype=int)

    def go_around_single_oxid_n_single_neigh(self, around_coords):
        """Does not distinguish between multiple flat neighbours. If at least one flat neighbour P=P1"""
        flat_neighbours = go_around_bool(self.cur_case.precip_3d_init, around_coords)
        temp = np.array([np.sum(item) for item in flat_neighbours], dtype=bool)

        return np.array(temp, dtype=np.ubyte)

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
