from math import *


DENSITY = {"Ni": 8908, "Fe": 7800, "Ti": 4506, "Cr": 7140, "Al": 2700, "N": 1.17, "O": 1, "None": 1}  # kg/m3
MOLAR_MASS = {"Ti": 0.0479, "Ni": 0.0587, "Fe": 0.0558, "Cr": 0.052, "Al": 0.027, "N": 0.014, "O": 0.016, "TiN": 0.0619,
              "None": 1}  # kg/mol

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

    "neigh_range": 1}

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


def get_diff_coeff(temperature, cond):
    diff_coeff = {"N in Ni20Cr2Ti Krupp": 4.7 * 10 ** -6 * exp(-125720 / (8.314 * (273 + temperature))),
                  "O in Ni at 1000 Smithells_Ransley": 2.4 * 10**-13,
                  "O in Ni Krupp": 4.9 * 10 ** -6 * exp(-164000 / (8.314 * (273 + temperature))),
                  "N in Ni Krupp": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "Al in Ni Krupp": 1.85 * 10 ** -4 * exp(-260780 / (8.314 * (273 + temperature))),
                  "Cr in Ni Krupp": 5.2 * 10 ** -4 * exp(-289000 / (8.314 * (273 + temperature))),
                  "Ti in Ni Krupp": 4.1 * 10 ** -4 * exp(-275000 / (8.314 * (273 + temperature))),
                  "N in Ni Katrin PHD": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "N in Ni Savva at 1020": 5 * 10 ** -11,
                  "Ti in Ni Savva at 1020": 5 * 10 ** -15,
                  "N in alfa-Fe Rozendaal": 6.6 * 10 ** -7 * exp(-77900 / (8.314 * (273 + temperature))),
                  "Test": 1 * (10 ** -15),
                  "Test_slower": 1 * (10 ** -14),
                  "Ti in Ni Krupp boost": 2.13 * 10 ** -13,
                  "Test Diffusion in precipitation": 6.18 * 10 ** -20,
                  "N in FeAlMn": 2.1 * 10 ** -8 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Al in FeAlMn": 2.1 * 10 ** -10 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Cr in NiCr": 0.03 * exp(-40800/(8.314 * (273 + temperature))),
                  "N in NiCr at 800°C": 6.7 * 10 ** -11
                  }
    return diff_coeff[cond]
