import copy
from engine import *
import traceback
import elements


if __name__ == '__main__':

    user_input = {"oxidant": {"primary": {"elem": "O",
                                          "diffusion_condition": "O in Ni Krupp",
                                          "cells_concentration": 0.1},
                              "secondary": {"elem": "None",
                                            "diffusion_condition": "N in Ni Krupp",
                                            "cells_concentration": 0.3}
                              },

                  "active_element": {"primary": {"elem": "Al",
                                                 "diffusion_condition": "O in Ni Krupp",
                                                 "mass_concentration": 0.03,
                                                 "cells_concentration": 0.1},
                                     "secondary": {"elem": "None",
                                                   "diffusion_condition": "Al in Ni Krupp",
                                                   "mass_concentration": 0.0,
                                                   "cells_concentration": 0.0}
                                     },

                  "matrix_elem": {"elem": "Ni",
                                  "diffusion_condition": "not_used",
                                  "concentration": 0},

                  "full_cells": False,
                  "diff_in_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
                  "diff_out_precipitation": 3.05 * 10 ** -14,  # [m2/sek]
                  "temperature": 1100,  # °C
                  "n_cells_per_axis": 102,  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
                  "n_iterations": 300000,  # must be >= n_cells_per_axis
                  "stride": 1,  # n_iterations / stride = n_iterations for outward diffusion
                  "sim_time": 72000,  # [sek]
                  "size": 70 * (10**-6),  # [m]

                  "threshold_inward": 1,
                  "threshold_outward": 1,
                  "sol_prod": 6.25 * 10 ** -31,  # 5.621 * 10 ** -10

                  "nucleation_probability": 0,
                  "het_factor": 10**0.5,  # not used anymore

                  "dissolution_p": 1 * 10 ** 0,
                  "dissolution_n": 2,  # not used anymore
                  "exponent_power": 0,  # not used anymore
                  "block_scale_factor": 1,

                  "inward_diffusion": True,
                  "outward_diffusion": True,
                  "compute_precipitations": True,
                  "diffusion_in_precipitation": False,

                  "save_whole": False,
                  "save_path": 'C:/test_runs_data/',

                  "neigh_range": 1,  # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                                     #          and           |  |  |  |  |   |   |   |   |   |
                                     # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
                  "decompose_precip": True,

                  "phase_fraction_lim": 0.6,
                  "hf_deg_lim": 10**10,
                  "lowest_neigh_numb": 16,
                  "final_nucl_prob": 1*10**0,

                  "min_dissol_prob": 1 * 10 ** -2.4,
                  "het_factor_dissolution": 10 ** 1,  # not used anymore
                  "final_dissol_prob": 1 * 10 ** 0,
                  "final_het_factor_dissol": 10 ** 0,  # not used anymore
                  "final_min_dissol_prob": 1 * 10 ** -1,

                  "max_neigh_numb": 0,
                  "product_kinetic_const": 0.0000003,  # not used anymore
                  "error_prod_conc": 1.01,  # not used anymore

                  "init_P1": 0.9999,
                  "final_P1": 1 * 10 ** 0,
                  "b_const_P1": 1,

                  "nucl_adapt_function": 3,
                  "dissol_adapt_function": 5,

                  "init_P1_diss": 0.9999,
                  "final_P1_diss": 1 * 10 ** 0,
                  "b_const_P1_diss": 1,

                  "b_const_P0_nucl": 1,

                  "bend_b_init": 0.5,
                  "bend_b_final": 0.1,

                  }

    backup_user_input = copy.deepcopy(user_input)
    eng = CellularAutomata(user_input=user_input)

    # set oxidant
    primary_oxidant = elements.OxidantElem(eng.param["oxidant"]["primary"], eng.utils)
    primary_oxidant.diffuse = primary_oxidant.diffuse_with_scale
    n_per_page = primary_oxidant.n_per_page
    primary_oxidant.cells = np.random.randint(user_input["n_cells_per_axis"],
                                              size=(3, int(n_per_page * user_input["n_cells_per_axis"])),
                                              dtype=np.short)
    primary_oxidant.dirs = np.random.choice([22, 4, 16, 10, 14, 12], len(primary_oxidant.cells[0]))
    primary_oxidant.dirs = np.array(np.unravel_index(primary_oxidant.dirs, (3, 3, 3)), dtype=np.byte)
    primary_oxidant.dirs -= 1
    eng.primary_oxidant = primary_oxidant
    eng.cases.first.oxidant = primary_oxidant
    eng.primary_oxidant.scale = eng.primary_product

    eng.primary_oxidant.transform_to_3d()

    # set active
    primary_active = elements.ActiveElem(eng.param["active_element"]["primary"])
    eng.primary_active = primary_active
    primary_active.diffuse = primary_active.diffuse_with_scale
    eng.cases.first.active = primary_active
    eng.primary_active.scale = eng.primary_product

    eng.primary_active.transform_to_3d(user_input["n_cells_per_axis"])

    # set product
    # first sphere
    half_thickness = 15
    middle = int(user_input["n_cells_per_axis"] / 2)
    minus = middle - half_thickness
    plus = middle + half_thickness
    shift = 15

    # for x in range(minus, plus):
    #     for y in range(minus, plus):
    #         for z in range(minus, plus):
    #             if (x - middle) ** 2 + (y - middle) ** 2 + (z - middle) ** 2 <= half_thickness ** 2:
    #                 eng.primary_product.c3d[z, y, x + shift] = eng.primary_oxid_numb
    #                 eng.primary_product.full_c3d[z, y, x + shift] = True
    #                 eng.product_x_nzs[x + shift] = True
    #                 eng.primary_oxidant.c3d[z, y, x + shift] = 0
    #                 eng.primary_active.c3d[z, y, x + shift] = 0

    # second sphere
    half_thickness = 5
    shift = -20

    for x in range(minus, plus):
        for y in range(minus, plus):
            for z in range(minus, plus):
                if (x - middle) ** 2 + (y - middle) ** 2 + (z - middle) ** 2 <= half_thickness ** 2:
                    eng.primary_product.c3d[z, y, x + shift] = eng.primary_oxid_numb
                    eng.primary_product.full_c3d[z, y, x + shift] = True
                    eng.product_x_nzs[x + shift] = True
                    eng.primary_oxidant.c3d[z, y, x + shift] = 0
                    eng.primary_active.c3d[z, y, x + shift] = 0

    eng.primary_oxidant.transform_to_descards()
    eng.primary_active.transform_to_descards()

    # mid_point_coord = int((user_input["n_cells_per_axis"] - 1) / 2)
    # eng.product_x_nzs[mid_point_coord - 1:mid_point_coord + 2] = True

    # set functions
    eng.precip_func = eng.precipitation_growth_test_with_p1
    # eng.check_intersection = eng.ci_single_only_p1
    eng.check_intersection = eng.ci_single

    eng.decomposition = eng.dissolution_test  # the function which was caled after is self.dissolution_zhou_wei_no_bsf()!!!!

    if eng.primary_oxid_numb > 1:
        eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_single_neigh
    else:
        eng.cases.first.go_around_func_ref = eng.go_around_single_oxid_n_single_neigh

    # set case
    eng.cur_case = eng.cases.first

    # eng.product_indexes = np.full(user_input["n_cells_per_axis"], False, dtype=bool)
    # eng.product_indexes[minus:plus] = True
    #
    # eng.product_indexes = np.array(np.where(eng.product_indexes)[0])
    #
    # # set oxidant
    # primary_oxidant = elements.OxidantElem(eng.param["oxidant"]["primary"], eng.utils)
    # primary_oxidant.n_per_page = 0  # to disable the appearing of new oxidant cells at the left boundary
    #
    # eng.primary_oxidant = primary_oxidant
    # eng.cases.first.oxidant = primary_oxidant
    #
    # # set active
    # primary_active = elements.ActiveElem(eng.param["active_element"]["primary"])
    # eng.primary_active = primary_active
    # eng.cases.first.active = primary_active
    #
    # # set functions
    # eng.cur_case = eng.cases.first
    # eng.decomposition = eng.dissolution_zhou_wei_original

    try:
        eng.simulation()
    finally:
        try:
            if not user_input["save_whole"]:
                eng.save_results()

        except (Exception,):
            backup_user_input["save_path"] = "C:/test_runs_data/"
            eng.utils = Utils(backup_user_input)
            eng.utils.create_database()
            eng.utils.generate_param()
            eng.save_results()
            print()
            print("____________________________________________________________")
            print("Saving To Standard Folder Crashed!!!")
            print("Saved To ->> C:/test_runs_data/!!!")
            print("____________________________________________________________")
            print()

            # data = np.column_stack(
            #     (np.arange(eng.iteration), eng.cumul_prod[:eng.iteration]))
            # output_file_path = "C:/test_runs_data/" + eng.utils.param["db_id"] + ".txt"
            # with open(output_file_path, "w") as f:
            #     for row in data:
            #         f.write(" ".join(map(str, row)) + "\n")

        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()

    # conz_list = [0.25, 0.55, 0.6, 0.65, 0.75, 0.8, 0.85]
    # for conc in conz_list:
    #     backup_user_input = copy.deepcopy(user_input)
    #     backup_user_input["active_element"]["primary"]["cells_concentration"] = conc
    #
    #     eng = CellularAutomata(user_input=backup_user_input)
    #
    #     try:
    #         eng.simulation()
    #     finally:
    #         try:
    #             if not backup_user_input["save_whole"]:
    #                 eng.save_results()
    #
    #         except (Exception, ):
    #             backup_user_input["save_path"] = "C:/Users/aseregin/Safe_folder_if_server_crash/"
    #             eng.utils = Utils(backup_user_input)
    #             eng.utils.create_database()
    #             eng.utils.generate_param()
    #             eng.save_results()
    #             print()
    #             print("____________________________________________________________")
    #             print("Saving To Standard Folder Crashed!!!")
    #             print("Saved To ->> C:/Users/aseregin/Safe_folder_if_server_crash/!!!")
    #             print("____________________________________________________________")
    #             print()
    #
    #         # data = np.column_stack(
    #         #     (np.arange(eng.iteration), eng.cumul_prod[:eng.iteration]))
    #         # output_file_path = "W:/SIMCA/test_runs_data/" + eng.utils.param["db_id"] + ".txt"
    #         # with open(output_file_path, "w") as f:
    #         #     for row in data:
    #         #         f.write(" ".join(map(str, row)) + "\n")
    #
    #         eng.insert_last_it()
    #         eng.utils.db.conn.commit()
    #         print()
    #         print("____________________________________________________________")
    #         print("Simulation was closed at Iteration: ", eng.iteration)
    #         print("____________________________________________________________")
    #         print()
    #         traceback.print_exc()

    # hf_list = [-19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
    #
    # for hf in hf_list:
    #     temp_user_input = copy.deepcopy(user_input)
    #     temp_user_input["hf_deg_lim"] = 10**hf
    #
    #     eng = CellularAutomata(user_input=temp_user_input)
    #
    #     print(f"""!!!!!!!!!!!!! ----->>>>>>>>>>>>>>>>>>>>>>>>>  DB:{temp_user_input["save_path"]}, Hf: {temp_user_input["hf_deg_lim"]}  <<<<<<<<<<<<<<<<<<<<<<<<<<<------- """)
    #     try:
    #         eng.simulation()
    #     finally:
    #         if not user_input["save_whole"]:
    #             eng.save_results()
    #         eng.insert_last_it()
    #         eng.utils.db.conn.commit()
    #         print()
    #         print("____________________________________________________________")
    #         print("Simulation was closed at Iteration: ", eng.iteration)
    #         print("____________________________________________________________")
    #         print()
    #         traceback.print_exc()



# # inw_sec = [0.001, 0.002, 0.003, 0.004, 0.005, 0.1, 0.3, 0.6, 1]
    #
    # # outw_sec = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25,
    # #             0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # inw_sec = [1]
    # outw_sec = [0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # results = []
    #
    # for inw_conc in inw_sec:
    #     print()
    #     print("________________________________________________________________________________________________________")
    #     print(inw_conc)
    #     print()
    #     user_input2["oxidant"]["primary"]["cells_concentration"] = inw_conc
    #     for outw_conc in outw_sec:
    #         user_input2["active_element"]["primary"]["cells_concentration"] = outw_conc
    #
    #         temp_user_input = copy.deepcopy(user_input2)
    #
    #         for step in range(3):
    #             eng = CellularAutomata(user_input=temp_user_input)
    #             results.append(eng.simulation())
    #             temp_user_input = copy.deepcopy(user_input2)
    #
    #         summ = np.sum(results)
    #         aver = summ/3
    #         # print(f"Average numb of products for inw {inw_conc} outw {outw_conc}: {aver} ")
    #         print(aver, end=", ")
    #         results = []
