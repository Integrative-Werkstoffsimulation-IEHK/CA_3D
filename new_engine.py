import gc

from cellular_automata import *
from utils import data_base
import progressbar
import time


class SimulationConfigurator:
    def __init__(self):
        self.ca = CellularAutomata()
        self.db = data_base.Database()
        self.begin = None
        self.elapsed_time = None

    def configurate_functions(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_with_scale
        self.ca.primary_active.diffuse = self.ca.primary_active.diffuse_with_scale

        self.ca.precip_func = self.ca.precipitation_first_case_MP
        self.ca.get_combi_ind = self.ca.get_combi_ind_atomic_with_kinetic_and_KP
        self.ca.precip_step = self.ca.precip_step_standard_MP
        self.ca.check_intersection = self.ca.ci_single_MP

        self.ca.decomposition = self.ca.dissolution_atomic_with_kinetic_MP
        self.ca.decomposition_intrinsic = self.ca.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL_MP

        self.ca.cur_case = self.ca.cases.first
        # self.ca.cases.first.go_around_func_ref = self.ca.go_around_mult_oxid_n_also_partial_neigh_aip_MP

        self.ca.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                  Config.PRODUCTS.PRIMARY)
        self.ca.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                    Config.PRODUCTS.PRIMARY)

    def configurate_functions1(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_bulk
        self.ca.primary_active.diffuse = self.ca.primary_active.diffuse_bulk

        self.ca.precip_func = self.ca.precipitation_first_case
        self.ca.get_combi_ind = self.ca.get_combi_ind_standard
        self.ca.precip_step = self.ca.precip_step_standard
        self.ca.check_intersection = self.ca.ci_single

        self.ca.decomposition = self.ca.dissolution_atomic_with_kinetic_MP
        self.ca.decomposition_intrinsic = self.ca.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL_MP

        self.ca.cur_case = self.ca.cases.first
        # self.ca.cases.first.go_around_func_ref = self.ca.go_around_mult_oxid_n_also_partial_neigh_aip
        # self.ca.cases.first.fix_init_precip_func_ref = self.ca.fix_init_precip_dummy

        self.ca.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                  Config.PRODUCTS.PRIMARY)
        self.ca.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                    Config.PRODUCTS.PRIMARY)

    def configurate_functions2(self):
        self.ca.primary_oxidant.diffuse = self.ca.primary_oxidant.diffuse_with_scale
        self.ca.primary_active.diffuse = self.ca.primary_active.diffuse_with_scale

        self.ca.precip_func = self.ca.precipitation_first_case
        self.ca.get_combi_ind = self.ca.get_combi_ind_atomic_with_kinetic_and_KP
        self.ca.precip_step = self.ca.precip_step_standard
        self.ca.check_intersection = self.ca.ci_single

        self.ca.decomposition = self.ca.dissolution_atomic_with_kinetic_and_KP
        self.ca.decomposition_intrinsic = self.ca.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        self.ca.cur_case = self.ca.cases.first
        self.ca.cases.first.go_around_func_ref = self.ca.go_around_mult_oxid_n_also_partial_neigh_aip

        self.ca.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                  Config.PRODUCTS.PRIMARY)
        self.ca.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                    Config.PRODUCTS.PRIMARY)

    def run_simulation(self):
        self.begin = time.time()

        for self.ca.iteration in progressbar.progressbar(range(Config.N_ITERATIONS)):
            # if self.ca.iteration % self.ca.precipitation_stride == 0:
            #     self.enforce_gc()
            # if self.iteration % self.precipitation_stride == 0:
            self.ca.precip_func()
            self.ca.decomposition()
                # self.calc_precipitation_front_only_cells()
            # self.precip_func()
            # self.decomposition()
            self.ca.diffusion_inward()
            self.ca.diffusion_outward()
            # self.diffusion_outward_with_mult_srtide()

        end = time.time()
        self.elapsed_time = (end - self.begin)
        self.db.insert_time(self.elapsed_time)
        self.db.conn.commit()

        self.terminate_workers()

    def save_results(self):
        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_descards()
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_descards()
        if Config.INWARD_DIFFUSION:
            self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)
            if Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_oxidant", self.ca.iteration, self.ca.secondary_oxidant.cells)
        if Config.OUTWARD_DIFFUSION:
            self.db.insert_particle_data("primary_active", self.ca.iteration, self.ca.primary_active.cells)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_active", self.ca.iteration, self.ca.secondary_active.cells)
        if Config.COMPUTE_PRECIPITATION:
            self.db.insert_particle_data("primary_product", self.ca.iteration, self.ca.primary_product.transform_c3d())
            if Config.ACTIVES.SECONDARY_EXISTENCE and Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.ca.iteration,
                                                   self.ca.secondary_product.transform_c3d())
                self.db.insert_particle_data("ternary_product", self.ca.iteration,
                                                   self.ca.ternary_product.transform_c3d())
                self.db.insert_particle_data("quaternary_product", self.ca.iteration,
                                                   self.ca.quaternary_product.transform_c3d())
            elif Config.ACTIVES.SECONDARY_EXISTENCE and not Config.OXIDANTS.SECONDARY_EXISTENCE:
                self.db.insert_particle_data("secondary_product", self.ca.iteration,
                                                   self.ca.secondary_product.transform_c3d())
        if Config.STRIDE > Config.N_ITERATIONS:
            self.ca.primary_active.transform_to_3d(self.ca.curr_max_furthest)
            if Config.ACTIVES.SECONDARY_EXISTENCE:
                self.ca.secondary_active.transform_to_3d(self.ca.curr_max_furthest)

    def terminate_workers(self):
        # # Signal workers to terminate
        # for _ in self.ca.workers:
        #     self.ca.input_queue.put(None)
        #
        # # Wait for all workers to terminate
        # for worker in self.ca.workers:
        #     worker.join()

        self.ca.pool.close()
        self.ca.pool.join()

        self.ca.precip_3d_init_shm.close()
        self.ca.precip_3d_init_shm.unlink()

        self.ca.product_x_nzs_shm.close()
        self.ca.product_x_nzs_shm.unlink()

        self.ca.primary_active.c3d_shared.close()
        self.ca.primary_active.c3d_shared.unlink()

        self.ca.primary_oxidant.c3d_shared.close()
        self.ca.primary_oxidant.c3d_shared.unlink()

        self.ca.primary_product.c3d_shared.close()
        self.ca.primary_product.c3d_shared.unlink()

        self.ca.primary_product.full_c3d_shared.close()
        self.ca.primary_product.full_c3d_shared.unlink()

        print("TERMINATED AND UNLINKED PROPERLY!")

    def save_results_only_prod(self):
        self.db.insert_particle_data("primary_product", self.ca.iteration,
                                           self.ca.primary_product.transform_c3d())

    def save_results_prod_and_inw(self):
        self.db.insert_particle_data("primary_product", self.ca.iteration,
                                           self.ca.primary_product.transform_c3d())
        self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)

    def save_results_only_inw(self):
        self.db.insert_particle_data("primary_oxidant", self.ca.iteration, self.ca.primary_oxidant.cells)

    def insert_last_it(self):
        self.db.insert_last_iteration(self.ca.iteration)

    def enforce_gc(self):
        # for _ in self.ca.workers:
        #     args = "GC"
        #     self.ca.input_queue.put(args)
        #
        # results = []
        # for _ in self.ca.workers:
        #     result = self.ca.output_queue.get()
        #     results.append(result)

        args = [("GC") for _ in range(self.ca.numb_of_proc)]
        results = self.ca.pool.map(CellularAutomata.worker, args)
        gc.collect()
        print("Done GC. Results after: ", results)
