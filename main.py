import copy
from engine import *
import traceback

if __name__ == '__main__':

    user_input = {"oxidant": {"primary": {"elem": "O",
                                          "diffusion_condition": "O in Ni Krupp",
                                          "cells_concentration": 0.01},
                              "secondary": {"elem": "None",
                                            "diffusion_condition": "N in Ni Krupp",
                                            "cells_concentration": 0.01}
                              },

                  "active_element": {"primary": {"elem": "Al",
                                                 "diffusion_condition": "Al in Ni Krupp",
                                                 "mass_concentration": 0.025,
                                                 "cells_concentration": 0.3},
                                     "secondary": {"elem": "None",
                                                   "diffusion_condition": "Al in Ni Krupp",
                                                   "mass_concentration": 0.025,
                                                   "cells_concentration": 0.077037037}
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
                  "stride": 40,  # n_iterations / stride = n_iterations for outward diffusion
                  "sim_time": 72000,  # [sek]
                  "size": 500 * (10**-6),  # [m]

                  "threshold_inward": 1,
                  "threshold_outward": 1,
                  "sol_prod": 0,  # 5.621 * 10 ** -10

                  "nucleation_probability": 10**-1,
                  "het_factor": 10**1,

                  "dissolution_p": 10**-3,
                  "dissolution_n": 3,
                  "exponent_power": 0,  # not used anymore
                  "block_scale_factor": 1,

                  "inward_diffusion": True,
                  "outward_diffusion": True,
                  "compute_precipitations": True,
                  "diffusion_in_precipitation": False,

                  "save_whole": False,
                  "save_path": 'W:/SIMCA/test_runs_data/',

                  "neigh_range": 1,  # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                                     #          and           |  |  |  |  |   |   |   |   |   |
                                     # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
                  "decompose_precip": True,

                  "phase_fraction_lim": 0.5,
                  "hf_deg_lim": 10**-1,  # range 0 - 1

                  "lowest_neigh_numb": 1,
                  "final_nucl_prob": 10**-4
                  }

    eng = CellularAutomata(user_input=user_input)
    try:
        eng.simulation()
    finally:
        if not user_input["save_whole"]:
            eng.save_results()
        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()

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
