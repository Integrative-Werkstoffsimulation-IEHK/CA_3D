DEFAULT_PARAM = {
    "oxidant": {"primary": {"elem": "N",
                            "diffusion_condition": "N in Ni20Cr2Ti Krupp",
                            "cells_concentration": 0.01},
                "secondary": {"elem": "None",
                              "diffusion_condition": "Test",
                              "cells_concentration": 0.1}},

    "active_element": {"primary": {"elem": "Ti",
                                   "diffusion_condition": "Ti in Ni Krupp",
                                   "mass_concentration": 0.02,
                                   "cells_concentration": 0.04},
                       "secondary": {"elem": "None",
                                     "diffusion_condition": "Test",
                                     "mass_concentration": 0.02,
                                     "cells_concentration": 0.02}
                       },
    "matrix_elem": {"elem": "Ni",
                    "diffusion_condition": "not_used",
                    "concentration": 0},

    "full_cells": False,
    "diff_in_precipitation": 3.05 * 10 ** -14,
    "diff_out_precipitation": 3.05 * 10 ** -14,
    "temperature": 1000,
    "n_cells_per_axis": 102,
    "n_iterations": 1000,
    "stride": 1,
    "sim_time": 720000,
    "size": 0.0005,

    "threshold_inward": 1,
    "threshold_outward": 1,
    "sol_prod": 5.621 * 10 ** -10,

    "nucleation_probability": 1,
    "het_factor": 300,

    "dissolution_p": 0.1,
    "dissolution_n": 2,
    "exponent_power": 3,
    "block_scale_factor": 2,

    "inward_diffusion": True,
    "outward_diffusion": True,
    "compute_precipitations": True,
    "diffusion_in_precipitation": None,

    "save_whole": False,
    "save_path": 'W:/SIMCA/test_runs_data/',

    "neigh_range": 1,
    "decompose_precip": False}

DEFAULT_OBJ_REF = {
    0: {"oxidant": None,
        "active": None,
        "product": None,
        "to_check_with": None},
    1: {"oxidant": None,
        "active": None,
        "product": None,
        "to_check_with": None},
    2: {"oxidant": None,
        "active": None,
        "product": None,
        "to_check_with": None},
    3: {"oxidant": None,
        "active": None,
        "product": None,
        "to_check_with": None}
                   }

TD_LOOKUP = {}
