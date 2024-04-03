from . import physical_data
from . import data_base
from . import templates
import sys
import numpy as np
from math import *
from . import probabilities


class Utils:
    def __init__(self, user_input):
        self.user_input = user_input
        self.param = self.user_input
        self.db = None
        self.n_cells_per_axis = user_input["n_cells_per_axis"]

        self.ind_decompose = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],   # 5 flat
             [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],      # 9  corners
             [-1, 1, -1], [-1, 1, 1], [-1, -1, -1], [-1, -1, 1],  # 13
             [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],  # 19 side corners
             [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_decompose_flat = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.byte)

        if self.user_input["neigh_range"] > 1:
            self.ind_formation = self.generate_neigh_indexes()
        else:
            self.ind_formation = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0],
                 [1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
                 [-1, 1, -1], [-1, 1, 1], [-1, -1, -1],
                 [-1, -1, 1], [1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0], [0, 1, -1], [0, 1, 1],
                 [0, -1, -1], [0, -1, 1], [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]], dtype=np.byte)

        self.ind_formation_noz = np.array(np.delete(self.ind_formation, 13, 0), dtype=np.byte)

        self.interface_neigh = {(0, 0, 1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
                                (0, 0, -1): [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]],
                                (0, 1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0]],
                                (0, -1, 0): [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]],
                                (1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0]],
                                (-1, 0, 0): [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [-1, 0, 0]]}

        # self.ind_excl_side = [[1, 1, -1], [1, 1, 0], [1, 1, 1], [1, 0, -1], [1, 0, 1],
        #                       [1, -1, -1], [1, -1, 0], [1, -1, 1], [0, 1, -1], [0, 1, 1],
        #                       [0, -1, -1], [0, -1, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
        #                       [-1, 0, -1],  [-1, 0, 1], [-1, -1, -1], [-1, -1, 0], [-1, -1, 1]]
        # self.ind_growth = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # self.ind_corners = [[1, 1, -1], [1, 1, 1], [1, -1, -1], [1, -1, 1],
        #                     [-1, 1, -1], [-1, 1, 1],  [-1, -1, -1], [-1, -1, 1]],
        # self.ind_side_corners = [[1, 1, 0], [1, 0, -1], [1, 0, 1], [1, -1, 0],
        #                          [0, 1, -1], [0, 1, 1], [0, -1, -1], [0, -1, 1],
        #                          [-1, 1, 0], [-1, 0, -1], [-1, 0, 1], [-1, -1, 0]]
        # self.ind_0 = [[1, 1, -1], [1, 1, 0], [1, 1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, -1, -1], [1, -1, 0],
        #               [1, -1, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, -1, -1],
        #               [0, -1, 0], [0, -1, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        #               [-1, -1, -1], [-1, -1, 0], [-1, -1, 1]]
        # self.ind = [[1, 1, -1], [1, 1, 0], [1, 1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, -1, -1], [1, -1, 0],
        #             [1, -1, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [0, 0, -1], [0, 0, 1], [0, -1, -1],
        #             [0, -1, 0], [0, -1, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        #             [-1, -1, -1], [-1, -1, 0], [-1, -1, 1]]

    def create_database(self):
        self.db = data_base.Database(self.user_input)

    def generate_param(self):
        # self.param = self.user_input
        self.param["tau"] = self.param["sim_time"] / self.param["n_iterations"]
        self.param["l_ambda"] = self.param["size"] / self.param["n_cells_per_axis"]
        self.param["dissolution_pn"] = self.param["dissolution_p"] ** (1 / self.param["dissolution_n"])

        if self.param["active_element"]["secondary"]["elem"] == "None":
            self.param["active_element"]["secondary"]["mass_concentration"] = 0
            self.param["active_element"]["secondary"]["cells_concentration"] = 0
            self.param[f"secondary_active_element_exists"] = False
        else:
            self.param[f"secondary_active_element_exists"] = True

        if self.param["oxidant"]["secondary"]["elem"] == "None":
            self.param[f"secondary_oxidant_exists"] = False
            self.param["oxidant"]["secondary"]["cells_concentration"] = 0
        else:
            self.param[f"secondary_oxidant_exists"] = True

        self.fetch_from_physical_data()
        self.calc_atomic_conc()
        self.check_c_min_and_calc_ncells()
        self.calc_active_data()
        self.calc_oxidant_data()
        self.calc_product_data()
        self.calc_initial_conc_and_moles()

    def calc_product_data(self):
        self.param["product"] = {"primary": {}, "secondary": {}, "ternary": {}, "quaternary": {}}

        # Primary
        self.param["product"]["primary"]["mass_per_cell"] = self.param["oxidant"]["primary"]["mass_per_cell"] + \
                                                            self.param["active_element"]["primary"]["mass_per_cell"]
        self.param["product"]["primary"]["moles_per_cell"] = self.param["active_element"]["primary"]["moles_per_cell"]/2
        self.param["product"]["primary"]["constitution"] = self.param["active_element"]["primary"]["elem"] + "+" +\
                                                        self.param["oxidant"]["primary"]["elem"]
        # Secondary
        self.param["product"]["secondary"]["mass_per_cell"] = self.param["oxidant"]["primary"]["mass_per_cell"] + \
                                                              self.param["active_element"]["secondary"]["mass_per_cell"]
        self.param["product"]["secondary"]["moles_per_cell"] = self.param["active_element"]["secondary"]["moles_per_cell"]/2
        self.param["product"]["secondary"]["constitution"] = self.param["active_element"]["secondary"]["elem"] + "+" + \
                                                           self.param["oxidant"]["primary"]["elem"]
        # Ternary
        self.param["product"]["ternary"]["mass_per_cell"] = self.param["oxidant"]["secondary"]["mass_per_cell"] + \
                                                              self.param["active_element"]["primary"]["mass_per_cell"]
        self.param["product"]["ternary"]["moles_per_cell"] = self.param["active_element"]["primary"]["moles_per_cell"]
        self.param["product"]["ternary"]["constitution"] = self.param["active_element"]["primary"]["elem"] + "+" + \
                                                             self.param["oxidant"]["secondary"]["elem"]
        # Quaternary
        self.param["product"]["quaternary"]["mass_per_cell"] = self.param["oxidant"]["secondary"]["mass_per_cell"] + \
                                                            self.param["active_element"]["secondary"]["mass_per_cell"]
        self.param["product"]["quaternary"]["moles_per_cell"] = self.param["active_element"]["secondary"]["moles_per_cell"]
        self.param["product"]["quaternary"]["constitution"] = self.param["active_element"]["secondary"]["elem"] + "+" + \
                                                           self.param["oxidant"]["secondary"]["elem"]

        t_1 = self.param["active_element"]["primary"]["molar_mass"] * self.param["matrix_elem"]["density"] / \
              (self.param["active_element"]["primary"]["density"] * self.param["matrix_elem"]["molar_mass"])
        t_2 = self.param["active_element"]["secondary"]["molar_mass"] * self.param["matrix_elem"]["density"] / \
              (self.param["active_element"]["secondary"]["density"] * self.param["matrix_elem"]["molar_mass"])

        self.param["product"]["primary"]["oxidation_number"] =\
            round(self.param["matrix_elem"]["moles_per_cell"] / (self.param["active_element"]["primary"]["moles_per_cell"]
                                                               * t_1))
        if self.param["product"]["primary"]["oxidation_number"] == 1:
            self.param["product"]["primary"]["lind_flat_arr"] = 6
        else:
            self.param["product"]["primary"]["lind_flat_arr"] = 7

        if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
            self.param["product"]["secondary"]["oxidation_number"] = \
                round(self.param["matrix_elem"]["moles_per_cell"] / (self.param["active_element"]["secondary"]["moles_per_cell"]
                                                                     * t_2))
            if self.param["product"]["secondary"]["oxidation_number"] == 1:
                self.param["product"]["secondary"]["lind_flat_arr"] = 6
            else:
                self.param["product"]["secondary"]["lind_flat_arr"] = 7

            self.param["product"]["ternary"]["oxidation_number"] = \
                round(self.param["matrix_elem"]["moles_per_cell"] / (self.param["active_element"]["primary"]["moles_per_cell"]
                                                                     * t_1))
            if self.param["product"]["ternary"]["oxidation_number"] == 1:
                self.param["product"]["ternary"]["lind_flat_arr"] = 6
            else:
                self.param["product"]["ternary"]["lind_flat_arr"] = 7

            self.param["product"]["quaternary"]["oxidation_number"] = \
                round(self.param["matrix_elem"]["moles_per_cell"] / (self.param["active_element"]["secondary"]["moles_per_cell"]
                                                                     * t_2))
            if self.param["product"]["quaternary"]["oxidation_number"] == 1:
                self.param["product"]["quaternary"]["lind_flat_arr"] = 6
            else:
                self.param["product"]["quaternary"]["lind_flat_arr"] = 7

        elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
            self.param["product"]["secondary"]["oxidation_number"] = \
                round(self.param["matrix_elem"]["moles_per_cell"] / (self.param["active_element"]["secondary"]["moles_per_cell"]
                                                                   * t_2))
            if self.param["product"]["secondary"]["oxidation_number"] == 1:
                self.param["product"]["secondary"]["lind_flat_arr"] = 6
            else:
                self.param["product"]["secondary"]["lind_flat_arr"] = 7

        self.param["product"]["primary"]["cells_per_axis"] = self.param["n_cells_per_axis"]
        self.param["product"]["secondary"]["cells_per_axis"] = self.param["n_cells_per_axis"]
        self.param["product"]["ternary"]["cells_per_axis"] = self.param["n_cells_per_axis"]
        self.param["product"]["quaternary"]["cells_per_axis"] = self.param["n_cells_per_axis"]

    def calc_oxidant_data(self):
        diff_coeff = self.param["oxidant"]["primary"]["diffusion_coefficient"]
        probabilities = self.calc_prob(diff_coeff)
        self.param["oxidant"]["primary"]["probabilities"] = probabilities
        self.param["oxidant"]["primary"]["p0_2d"] = self.calc_p0_2d(diff_coeff * 10**2)

        diff_coeff = self.param["oxidant"]["secondary"]["diffusion_coefficient"]
        probabilities = self.calc_prob(diff_coeff)
        self.param["oxidant"]["secondary"]["probabilities"] = probabilities
        self.param["oxidant"]["secondary"]["p0_2d"] = self.calc_p0_2d(diff_coeff)


        self.param["oxidant"]["primary"]["n_per_page"] = round(self.param["oxidant"]["primary"]["cells_concentration"] *
                                                             self.param["n_cells_per_axis"] ** 2)
        self.param["oxidant"]["secondary"]["n_per_page"] = round(self.param["oxidant"]["secondary"]["cells_concentration"] *
                                                               self.param["n_cells_per_axis"] ** 2)

        self.param["oxidant"]["primary"]["moles_per_cell"] = self.param["active_element"]["primary"]["moles_per_cell"] *\
                                                             1.5 / self.param["threshold_inward"]
        self.param["oxidant"]["primary"]["mass_per_cell"] = self.param["oxidant"]["primary"]["moles_per_cell"] *\
                                                            self.param["oxidant"]["primary"]["molar_mass"]

        self.param["oxidant"]["secondary"]["moles_per_cell"] = self.param["active_element"]["secondary"]["moles_per_cell"] / \
                                                             self.param["threshold_inward"]
        self.param["oxidant"]["secondary"]["mass_per_cell"] = self.param["oxidant"]["secondary"]["moles_per_cell"] * \
                                                            self.param["oxidant"]["secondary"]["molar_mass"]

    def fetch_from_physical_data(self):
        for elem_type in ["active_element", "oxidant"]:
            for elem_signature in ["primary", "secondary"]:
                self.param[elem_type][elem_signature]["cells_per_axis"] = self.param["n_cells_per_axis"]
                self.param[elem_type][elem_signature]["neigh_range"] = self.param["neigh_range"]
                elem = self.param[elem_type][elem_signature]["elem"]
                density = physical_data.DENSITY[elem]
                molar_mass = physical_data.MOLAR_MASS[elem]
                self.param[elem_type][elem_signature]["density"] = density
                self.param[elem_type][elem_signature]["molar_mass"] = molar_mass

                condition = self.param[elem_type][elem_signature]["diffusion_condition"]
                diff_coeff = physical_data.get_diff_coeff(self.param["temperature"], condition)
                self.param[elem_type][elem_signature]["diffusion_coefficient"] = diff_coeff

        elem = self.param["matrix_elem"]["elem"]
        density = physical_data.DENSITY[elem]
        molar_mass = physical_data.MOLAR_MASS[elem]
        cell_volume = self.param["l_ambda"] ** 3
        moles_per_cell = density * cell_volume / molar_mass
        mass = moles_per_cell * molar_mass
        self.param["matrix_elem"]["density"] = density
        self.param["matrix_elem"]["molar_mass"] = molar_mass
        self.param["matrix_elem"]["moles_per_cell"] = moles_per_cell
        self.param["matrix_elem"]["mass_per_cell"] = mass

    def calc_active_data(self):
        diff_coeff = self.param["active_element"]["primary"]["diffusion_coefficient"]
        probabilities = self.calc_prob(diff_coeff, stridden=True)
        self.param["active_element"]["primary"]["probabilities"] = probabilities

        diff_coeff = self.param["active_element"]["secondary"]["diffusion_coefficient"]
        probabilities = self.calc_prob(diff_coeff, stridden=True)
        self.param["active_element"]["secondary"]["probabilities"] = probabilities

        matrix_moles = self.param["matrix_elem"]["moles_per_cell"]
        matrix_molar_mass = self.param["matrix_elem"]["molar_mass"]

        atomic_c_1 = self.param["active_element"]["primary"]["atomic_concentration"]
        atomic_c_2 = self.param["active_element"]["secondary"]["atomic_concentration"]

        cells_conc1 = self.param["active_element"]["primary"]["cells_concentration"]
        cells_conc2 = self.param["active_element"]["secondary"]["cells_concentration"]

        molar_mass1 = self.param["active_element"]["primary"]["molar_mass"]
        molar_mass2 = self.param["active_element"]["secondary"]["molar_mass"]

        t_1 = self.param["active_element"]["primary"]["molar_mass"] * self.param["matrix_elem"]["density"] / \
              (self.param["active_element"]["primary"]["density"] * self.param["matrix_elem"]["molar_mass"])
        self.param["active_element"]["primary"]["n_ELEM"] = 1 - t_1
        t_2 = self.param["active_element"]["secondary"]["molar_mass"] * self.param["matrix_elem"]["density"] / \
              (self.param["active_element"]["secondary"]["density"] * self.param["matrix_elem"]["molar_mass"])
        self.param["active_element"]["secondary"]["n_ELEM"] = 1 - t_2
        denom = 1 + atomic_c_1 * (t_1 - 1) + atomic_c_2 * (t_2 - 1)

        moles_per_cell1 = atomic_c_1 * matrix_moles / (cells_conc1 * denom)
        self.param["active_element"]["primary"]["moles_per_cell"] = moles_per_cell1
        self.param["active_element"]["primary"]["mass_per_cell"] = moles_per_cell1 * molar_mass1
        self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"] = moles_per_cell1 * t_1
        self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"] = \
            self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"] * matrix_molar_mass

        if cells_conc2 > 0:
            moles_per_cell2 = atomic_c_2 * matrix_moles / (cells_conc2 * denom)
        else:
            moles_per_cell2 = 0
        self.param["active_element"]["secondary"]["moles_per_cell"] = moles_per_cell2
        self.param["active_element"]["secondary"]["mass_per_cell"] = moles_per_cell2 * molar_mass2
        self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"] = moles_per_cell2 * t_2
        self.param["active_element"]["secondary"]["eq_matrix_mass_per_cell"] = \
            self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"] * matrix_molar_mass

    def check_c_min_and_calc_ncells(self):
        t_1 = self.param["active_element"]["primary"]["molar_mass"] * self.param["matrix_elem"]["density"] /\
               (self.param["active_element"]["primary"]["density"] * self.param["matrix_elem"]["molar_mass"])
        t_2 = self.param["active_element"]["secondary"]["molar_mass"] * self.param["matrix_elem"]["density"] / \
              (self.param["active_element"]["secondary"]["density"] * self.param["matrix_elem"]["molar_mass"])
        atomic_c_1 = self.param["active_element"]["primary"]["atomic_concentration"]
        atomic_c_2 = self.param["active_element"]["secondary"]["atomic_concentration"]
        denom = 1 + atomic_c_1 * (t_1 - 1) + atomic_c_2 * (t_2 - 1)

        min_cells1 = atomic_c_1 * t_1 / denom
        if self.param["full_cells"]:
            self.param["active_element"]["primary"]["cells_concentration"] = min_cells1
            self.param["active_element"]["primary"]["n_per_page"] = round(min_cells1 * self.param["n_cells_per_axis"]**2)
        elif self.param["active_element"]["primary"]["cells_concentration"] < min_cells1:
            print()
            print("______________________________________________________________")
            print("Cells Concentration For primary active_element Must Be >= ", min_cells1, "!!!")
            print("______________________________________________________________")
            sys.exit()
        else:
            cells_conc = self.param["active_element"]["primary"]["cells_concentration"]
            self.param["active_element"]["primary"]["n_per_page"] = round(cells_conc * self.param["n_cells_per_axis"] ** 2)

        min_cells2 = atomic_c_2 * t_2 / denom
        if self.param["full_cells"]:
            if min_cells2 == 0:
                self.param["active_element"]["secondary"]["n_per_page"] = 0
            else:
                self.param["active_element"]["secondary"]["cells_concentration"] = min_cells2
                self.param["active_element"]["secondary"]["n_per_page"] = round(min_cells2 * self.param["n_cells_per_axis"] ** 2)
        elif self.param["active_element"]["secondary"]["cells_concentration"] < min_cells2:
            print()
            print("______________________________________________________________")
            print("Cells Concentration For secondary active_element Must Be >= ", min_cells2, "!!!")
            print("______________________________________________________________")
            sys.exit()
        else:
            cells_conc = self.param["active_element"]["secondary"]["cells_concentration"]
            self.param["active_element"]["secondary"]["n_per_page"] = round(cells_conc * self.param["n_cells_per_axis"] ** 2)

    def calc_atomic_conc(self):
        molar_mass1 = self.param["active_element"]["primary"]["molar_mass"]
        molar_mass2 = self.param["active_element"]["secondary"]["molar_mass"]
        molar_mass_matr = self.param["matrix_elem"]["molar_mass"]
        mass_conc1 = self.param["active_element"]["primary"]["mass_concentration"]
        mass_conc2 = self.param["active_element"]["secondary"]["mass_concentration"]

        self.param["active_element"]["primary"]["atomic_concentration"] =\
            mass_conc1 / (mass_conc1 + (molar_mass1 / molar_mass2) * mass_conc2 + (molar_mass1 / molar_mass_matr) *
                          (1 - mass_conc1 - mass_conc2))

        self.param["active_element"]["secondary"]["atomic_concentration"] = \
            mass_conc2 / (mass_conc2 + (molar_mass2 / molar_mass1) * mass_conc1 + (molar_mass2 / molar_mass_matr) *
                          (1 - mass_conc1 - mass_conc2))

    def calc_initial_conc_and_moles(self):
        inward = self.param["oxidant"]["primary"]["n_per_page"]
        inward_moles = inward * self.param["oxidant"]["primary"]["moles_per_cell"]
        inward_mass = inward * self.param["oxidant"]["primary"]["mass_per_cell"]

        sinward = self.param["oxidant"]["secondary"]["n_per_page"]
        sinward_moles = sinward * self.param["oxidant"]["secondary"]["moles_per_cell"]
        sinward_mass = sinward * self.param["oxidant"]["secondary"]["mass_per_cell"]

        outward = self.param["active_element"]["primary"]["n_per_page"]
        outward_moles = outward * self.param["active_element"]["primary"]["moles_per_cell"]
        outward_mass = outward * self.param["active_element"]["primary"]["mass_per_cell"]
        outward_eq_mat_moles = outward * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]

        soutward = self.param["active_element"]["secondary"]["n_per_page"]
        soutward_moles = soutward * self.param["active_element"]["secondary"]["moles_per_cell"]
        soutward_mass = soutward * self.param["active_element"]["secondary"]["mass_per_cell"]
        soutward_eq_mat_moles = soutward * self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"]

        n_cells_page = (self.n_cells_per_axis ** 2)
        matrix_moles = n_cells_page * self.param["matrix_elem"]["moles_per_cell"] - outward_eq_mat_moles \
                       - soutward_eq_mat_moles
        matrix_mass = matrix_moles * self.param["matrix_elem"]["molar_mass"]

        whole_moles = matrix_moles + inward_moles + sinward_moles + outward_moles + soutward_moles
        whole_mass = matrix_mass + inward_mass + sinward_mass + outward_mass + soutward_mass

        inward_c_moles = inward_moles / whole_moles
        sinward_c_moles = sinward_moles / whole_moles
        outward_c_moles = outward_moles / whole_moles
        soutward_c_moles = soutward_moles / whole_moles
        matrix_c_moles = matrix_moles / whole_moles

        inward_c_mass = inward_mass / whole_mass
        sinward_c_mass = sinward_mass / whole_mass
        outward_c_mass = outward_mass / whole_mass
        soutward_c_mass = soutward_mass / whole_mass
        matrix_c_mass = matrix_mass / whole_mass

        self.param["initial_conc_and_moles"] = {}
        self.param["initial_conc_and_moles"]["inward_moles"] = inward_moles
        self.param["initial_conc_and_moles"]["inward_mass"] = inward_mass

        self.param["initial_conc_and_moles"]["sinward_moles"] = sinward_moles
        self.param["initial_conc_and_moles"]["sinward_mass"] = sinward_mass

        self.param["initial_conc_and_moles"]["outward_moles"] = outward_moles
        self.param["initial_conc_and_moles"]["outward_mass"] = outward_mass
        self.param["initial_conc_and_moles"]["outward_eq_mat_moles"] = outward_eq_mat_moles

        self.param["initial_conc_and_moles"]["soutward_moles"] = soutward_moles
        self.param["initial_conc_and_moles"]["soutward_mass"] = soutward_mass
        self.param["initial_conc_and_moles"]["soutward_eq_mat_moles"] = soutward_eq_mat_moles

        self.param["initial_conc_and_moles"]["matrix_moles"] = matrix_moles
        self.param["initial_conc_and_moles"]["matrix_mass"] = matrix_mass

        self.param["initial_conc_and_moles"]["whole_moles"] = whole_moles
        self.param["initial_conc_and_moles"]["whole_mass"] = whole_mass

        self.param["initial_conc_and_moles"]["inward_c_moles"] = inward_c_moles
        self.param["initial_conc_and_moles"]["inward_c_mass"] = inward_c_mass

        self.param["initial_conc_and_moles"]["sinward_c_moles"] = sinward_c_moles
        self.param["initial_conc_and_moles"]["sinward_c_mass"] = sinward_c_mass

        self.param["initial_conc_and_moles"]["outward_c_moles"] = outward_c_moles
        self.param["initial_conc_and_moles"]["outward_c_mass"] = outward_c_mass

        self.param["initial_conc_and_moles"]["soutward_c_moles"] = soutward_c_moles
        self.param["initial_conc_and_moles"]["soutward_c_mass"] = soutward_c_mass

        self.param["initial_conc_and_moles"]["matrix_c_moles"] = matrix_c_moles
        self.param["initial_conc_and_moles"]["matrix_c_mass"] = matrix_c_mass

        if self.param["sol_prod"] != 0:
            self.param["initial_conc_and_moles"]["max_gamma_min_one"] = (((inward_c_moles ** 3) * (outward_c_moles ** 2))/
                                                                         self.param["sol_prod"]) - 1
        else:
            self.param["initial_conc_and_moles"]["max_gamma_min_one"] = 0

    def calc_prob(self, diff_coeff, stridden=False):
        if not stridden:
            coeff = 6 * (self.param["tau"] * diff_coeff) / (self.param["l_ambda"] ** 2)
        else:
            new_tau = self.param["sim_time"] / (self.param["n_iterations"] / self.param["stride"])
            coeff = 6 * (new_tau * diff_coeff) / (self.param["l_ambda"] ** 2)
        c = (1 - coeff) / (1 + coeff)
        if -(c + 1) / 8 > (c - 1) / 8:
            t = -(c + 1) / 16
        else:
            t = (c - 1) / 16
        p = -2 * t
        p3 = (1 / (coeff + 1)) - 2 * p
        p0 = 1 - 4 * p - p3
        return [p, p3, p0]

    def calc_p0_2d(self, diff_coeff):
        coeff = 4 * (self.param["tau"] * diff_coeff) / (self.param["l_ambda"] ** 2)
        coeff_p = coeff / (1 + coeff)

        # f = 1 - (1 / (2 * coeff_p))
        # g = 1 / coeff_p
        # h = 2 - (1 / coeff_p)
        #
        # print(1, " > t > ", f)
        # print(g, " > t > ", 0)
        # print(2, " > t > ", h)

        if coeff_p < 1:
            r_bound = 1
        else:
            r_bound = 1 / coeff_p

        l_bound = 0
        if 1 - (1/(2*coeff_p)) < 2 - (1/coeff_p):
            if 2 - (1/coeff_p) > 0:
                l_bound = 2 - (1/coeff_p)

        elif 1 - (1/(2*coeff_p)) > 2 - (1/coeff_p):
            if 1 - (1/(2*coeff_p)) > 0:
                l_bound = 1 - (1/(2*coeff_p))

        t = l_bound + (r_bound - l_bound) / 2

        p = coeff_p * (1 - t)
        p0 = t * coeff_p
        p3 = 1 + coeff_p * (t - 2)

        # d = (self.param["l_ambda"] ** 2) / (self.param["tau"]) * ((p + p0)/(4*(1-p-p0)))
        # one = 2*p + p0 + p3

        return p0

    def calc_prob_manually(self, p, diff_coeff):
        coeff = 6 * (self.param["tau"] * diff_coeff) / (self.param["l_ambda"] ** 2)
        p3 = (1 - 2 * p * (1 + coeff)) / (1 + coeff)
        p0 = 1 - 4 * p - p3
        return {"p": p, "p3": p3, "p0": p0}

    def calc_masses_solprod(self):
        m_matrix = physical_data.DENSITY[self.param["matrix_elem"]] * self.param["l_ambda"] ** 3
        m_act = self.param["active_elem_conc_real"] * m_matrix /\
                (self.param["active_elem_conc"] * (1 - self.param["active_elem_conc_real"]))
        a = self.param["sol_prod"] * self.param["threshold_inward"]**2
        b = self.param["threshold_inward"] * self.param["threshold_outward"] * (2 * self.param["sol_prod"] - 1) * m_act\
            + self.param["threshold_inward"] * 2 * self.param["sol_prod"] * m_matrix
        c = self.param["sol_prod"] * (self.param["threshold_outward"] * m_act + m_matrix)**2
        disk = (b**2 - 4 * a * c)**0.5
        two_a = 2 * a
        m_inward_1 = (-b + disk) / two_a
        m_inward_2 = (-b - disk) / two_a
        if m_inward_1 > m_inward_2:
            m_inward = m_inward_2
        else:
            m_inward = m_inward_1
        m_precip = m_act * self.param["threshold_outward"] + m_inward * self.param["threshold_inward"]
        return {"matrix": m_matrix, "active": m_act, "precipitation": m_precip, "inward": m_inward}

    def calc_sur_ind_decompose(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_decompose_flat(self, seeds):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        seeds = seeds.transpose()
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_decompose_flat] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        around_seeds[around_seeds == self.n_cells_per_axis] = 0
        around_seeds[around_seeds == -1] = self.n_cells_per_axis - 1
        return around_seeds

    def calc_sur_ind_formation(self, seeds, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dummy_ind:
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_formation] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        if seeds[0, 2] < self.param["neigh_range"]:
            indexes = np.where(around_seeds[:, :, 2] < 0)
            around_seeds[indexes[0], indexes[1], 2] = dummy_ind
        for shift in range(self.param["neigh_range"]):
            indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis + shift)
            around_seeds[indexes[0], indexes[1], indexes[2]] = shift
            indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
            around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - shift - 1
        return around_seeds

    def calc_sur_ind_formation_noz(self, seeds, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dummy_ind:
        :param seeds: seeds in descarts coordinates
        :return: around_seeds: array of the surrounding coordinates for each seed (26 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[item + self.ind_formation_noz] for item in seeds], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        if seeds[0, 2] < self.param["neigh_range"]:
            indexes = np.where(around_seeds[:, :, 2] < 0)
            around_seeds[indexes[0], indexes[1], 2] = dummy_ind
        for shift in range(self.param["neigh_range"]):
            indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis + shift)
            around_seeds[indexes[0], indexes[1], indexes[2]] = shift
            indexes = np.where(around_seeds[:, :, 0:2] == - shift - 1)
            around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - shift - 1
        return around_seeds

    def calc_sur_ind_interface(self, cells, dirs, dummy_ind):
        """
        Calculating the descarts surrounding coordinates for each seed including the position of the seed itself.
        :param dirs:
        :param cells:
        :param dummy_ind:
        :return: around_seeds: array of the surrounding coordinates for each seed (4 flat coordinates for each seed)
        """
        # generating a neighbouring coordinates for each seed (including the position of the seed itself)
        around_seeds = np.array([[cell + self.interface_neigh[tuple(dir)]]
                                 for cell, dir in zip(cells.transpose(), dirs.transpose())], dtype=np.short)[:, 0]
        # applying periodic boundary conditions
        indexes = np.where(around_seeds[:, :, 2] < 0)
        around_seeds[indexes[0], indexes[1], 2] = dummy_ind

        indexes = np.where(around_seeds[:, :, 0:2] == self.n_cells_per_axis)
        around_seeds[indexes[0], indexes[1], indexes[2]] = 0
        indexes = np.where(around_seeds[:, :, 0:2] == - 1)
        around_seeds[indexes[0], indexes[1], indexes[2]] = self.n_cells_per_axis - 1
        return around_seeds

    def generate_neigh_indexes(self):
        neigh_range = self.user_input["neigh_range"]
        size = 3 + (neigh_range - 1) * 2
        neigh_shape = (size, size, size)
        temp = np.ones(neigh_shape, dtype=int)
        coord = np.array(np.nonzero(temp))
        coord -= neigh_range
        coord = coord.transpose()
        return np.array(coord, dtype=np.byte)

    def print_init_var(self):
        expn = 2.71828182845904523536
        # nucl_prob = self.param["const_a"] * expn ** (self.param["const_b"] * np.array([1, 2, 3, 4, 5, 6]))
        # dissol_prob = self.param["dissolution_pn"] * \
        #               expn ** (-self.param["exponent_power"] * np.array([0, 1, 2, 3, 4, 5]))
        # dissol_p_block = dissol_prob[3] / self.param["block_scale_factor"]
        # dissol_prob_block = np.array([dissol_p_block, dissol_p_block / 10, dissol_p_block / 100])

        dissol_pn = self.param["dissolution_p"] ** (1 / self.param["dissolution_n"])
        power = (self.param["dissolution_n"] - 1) / self.param["dissolution_n"]
        const_b_dissol = (1 / 3) * log(self.param["block_scale_factor"] * (self.param["dissolution_p"] ** power))
        p_b_3 = dissol_pn * 2.718281828 ** (const_b_dissol * 3) / self.param["block_scale_factor"]
        p_b_2 = dissol_pn * 2.718281828 ** (const_b_dissol * 4) / (self.param["block_scale_factor"]**3)
        p_b_1 = dissol_pn * 2.718281828 ** (const_b_dissol * 5) / (self.param["block_scale_factor"]**10)

        dissol_prob = dissol_pn * expn ** (const_b_dissol * np.array([0, 1, 2, 3, 4, 5]))
        dissol_prob_block = np.array([p_b_3, p_b_2, p_b_1])

        print()
        print(f"""-------------------------------------------------------""")
        print(f"""DATA BASE AT: {self.param["save_path"]}""")
        print(f"""-------------------------------------------------------""", end="")
#         print(f"""
# SYSTEM PARAMETERS:----------------------------------------------------/PRECIPITATION:--------------------------------------------------------------------------------------
#             * Number of Cells Per Axis: {self.param["n_cells_per_axis"]:<30}/    * Solubility product: {self.param["sol_prod"]:<20}Zhou and Wei Parameters:
#             * Total Iterations: _______ {self.param["n_iterations"]:<30}/    * Threshold Inward:   {self.param["threshold_inward"]:<20}* p: {self.param["dissolution_p"]}
#             * Stirde: _________________ {self.param["stride"]:<30}/    * Threshold Outward:  {self.param["threshold_outward"]:<20}* n: {self.param["dissolution_n"]}
#             * Time [sek]: _____________ {self.param["sim_time"]:<30}/    * Neighbourhood distance: {self.param["neigh_range"]:<20}* block factor: {self.param["block_scale_factor"]}
#             * Time [h]: _______________ {self.param["sim_time"] / 3600:<30}/
#             * Length [m]: _____________ {self.param["size"]:<30}/    * Heterogeneous Factor:  {self.param["het_factor"]}
#                                                                       /    * Nucleation probability: {self.param["nucleation_probability"]}
#             Modules:                                                  /    * Neighbourhood distance: {self.param["neigh_range"]}
#             * inward_diffusion: _______ {bool(self.param["inward_diffusion"]):<30}/
#             * outward_diffusion: ______ {bool(self.param["outward_diffusion"]):<30}/    * Number of sides covered:              0       1       2        3        4        5       6
#             * compute_precipitations: _ {bool(self.param["compute_precipitations"]):<30}/    * Nucleation Probabilities:           {self.param["nucleation_probability"]}    {nucl_prob[0]:.2f}    {nucl_prob[1]:.2f}     {nucl_prob[2]:.2f}     {nucl_prob[3]:.2f}     {nucl_prob[4]:.2f}     1
#             * diffusion_in_precip: ____ {bool(self.param["diffusion_in_precipitation"]):<30}/    * Dissolution Probabilities:          {dissol_prob[0]:.4f}  {dissol_prob[1]:.4f}  {dissol_prob[2]:.4f}   {dissol_prob[3]:.4f}   {dissol_prob[4]:.4f}   {dissol_prob[5]:.4f}    0
#             * decompose_precip: _______ {bool(self.param["decompose_precip"]):<30}/    * Dissolution Probabilities in block:   -       -       -      {dissol_prob_block[0]:.4f}   {dissol_prob_block[1]:.4f}   {dissol_prob_block[2]:.4f}    -
#             * full_cells: _____________ {bool(self.param["full_cells"]):<30}
# """, end="")
        print(f"""
ELEMENTS:---------------------------------------------------------------------------------------------------------------------------------------------------------------
Primary Oxidant: {self.param["oxidant"]["primary"]["elem"]}                                                                        Primary Active: {self.param["active_element"]["primary"]["elem"]}
    * Concentration: {self.param["oxidant"]["primary"]["cells_concentration"]} [cells fraction]                                              * Concentrations: {self.param["active_element"]["primary"]["mass_concentration"]} [wt%]
    * Diffusion Condition: {self.param["oxidant"]["primary"]["diffusion_condition"]} => D= {self.param["oxidant"]["primary"]["diffusion_coefficient"]} [m^2/sek]                    {self.param["active_element"]["primary"]["atomic_concentration"]} [at%]
    * Probabilities: p: {self.param["oxidant"]["primary"]["probabilities"][0]}                                             * Diffusion Condition: {self.param["active_element"]["primary"]["diffusion_condition"]} => Coefficient:{self.param["active_element"]["primary"]["diffusion_coefficient"]} [m^2/sek]
                     p3: {self.param["oxidant"]["primary"]["probabilities"][1]}                                            * Probabilities: p: {self.param["active_element"]["primary"]["probabilities"][0]}
                     p0: {self.param["oxidant"]["primary"]["probabilities"][2]}                                                      p3: {self.param["active_element"]["primary"]["probabilities"][1]}
                     p0_2D: {self.param["oxidant"]["primary"]["p0_2d"]}
    * Moles per cell: {self.param["oxidant"]["primary"]["moles_per_cell"]} [mole]                                                   p0: {self.param["active_element"]["primary"]["probabilities"][2]}
    * Mass per cell: {self.param["oxidant"]["primary"]["mass_per_cell"]} [kg]                                           * Moles per cell: {self.param["active_element"]["primary"]["moles_per_cell"]} [mole]
    * Cells concentration: {self.param["oxidant"]["primary"]["cells_concentration"]}                                                         * Mass per cell: {self.param["active_element"]["primary"]["mass_per_cell"]} [kg]
    * N cells per page: {self.param["oxidant"]["primary"]["n_per_page"]}                                                                * Matrix Moles per cell: {self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]} [mole]
                                                                                          * Matrix Mass per cell: {self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]} [kg]
                                                                                          * Cells concentration: {self.param["active_element"]["primary"]["cells_concentration"]}
                                                                                          * N cells per page: {self.param["active_element"]["primary"]["n_per_page"]}
    --------------------------------------------------------------------------""", end="")
        if self.param["secondary_oxidant_exists"]:
            print(f"""
Secondary Oxidant: {self.param["oxidant"]["secondary"]["elem"]}
    * Concentration: {self.param["oxidant"]["secondary"]["cells_concentration"]} [cells fraction]
    * Diffusion Condition: {self.param["oxidant"]["secondary"]["diffusion_condition"]} => Coefficient:{self.param["oxidant"]["secondary"]["diffusion_coefficient"]} [m^2/sek]
    * Probabilities: p: {self.param["oxidant"]["secondary"]["probabilities"][0]}
                     p3: {self.param["oxidant"]["secondary"]["probabilities"][1]}
                     p0: {self.param["oxidant"]["secondary"]["probabilities"][2]}
    * Moles per cell: {self.param["oxidant"]["secondary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["oxidant"]["secondary"]["mass_per_cell"]} [kg]
    * Cells concentration: {self.param["oxidant"]["secondary"]["cells_concentration"]}
    * N cells per page: {self.param["oxidant"]["secondary"]["n_per_page"]}
    --------------------------------------------------------------------------""", end="")
        if self.param["secondary_active_element_exists"]:
            print(f"""
Secondary Active: {self.param["active_element"]["secondary"]["elem"]}
    * Concentration: {self.param["active_element"]["secondary"]["mass_concentration"]} [wt%]
                     {self.param["active_element"]["secondary"]["atomic_concentration"]} [at%]
    * Diffusion Condition: {self.param["active_element"]["secondary"]["diffusion_condition"]} => Coefficient:{self.param["active_element"]["secondary"]["diffusion_coefficient"]} [m^2/sek]
    * Probabilities: p: {self.param["active_element"]["secondary"]["probabilities"][0]}
                     p3: {self.param["active_element"]["secondary"]["probabilities"][1]}
                     p0: {self.param["active_element"]["secondary"]["probabilities"][2]}
    * Moles per cell: {self.param["active_element"]["secondary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["active_element"]["secondary"]["mass_per_cell"]} [kg]
    * Matrix Moles per cell: {self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"]} [mole]
    * Matrix Mass per cell: {self.param["active_element"]["secondary"]["eq_matrix_mass_per_cell"]} [kg]
    * Cells concentration: {self.param["active_element"]["secondary"]["cells_concentration"]}
    * N cells per page: {self.param["active_element"]["secondary"]["n_per_page"]}
    --------------------------------------------------------------------------""", end="")
        print(f"""
Primary Product: {self.param["active_element"]["primary"]["elem"]} + {self.param["oxidant"]["primary"]["elem"]}
    * Moles per cell: {self.param["product"]["primary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["product"]["primary"]["mass_per_cell"]} [kg]
    * Oxidation number: {self.param["product"]["primary"]["oxidation_number"]}
    --------------------------------------------------------------------------""")
        if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
            print(f"""
Secondary Product: {self.param["active_element"]["secondary"]["elem"]} + {self.param["oxidant"]["primary"]["elem"]}
    * Moles per cell: {self.param["product"]["secondary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["product"]["secondary"]["mass_per_cell"]} [kg]
    * Oxidation number: {self.param["product"]["secondary"]["oxidation_number"]}
    --------------------------------------------------------------------------""", end="")
            print(f"""
Ternary Product: {self.param["active_element"]["primary"]["elem"]} + {self.param["oxidant"]["secondary"]["elem"]}
    * Moles per cell: {self.param["product"]["ternary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["product"]["ternary"]["mass_per_cell"]} [kg]
    * Oxidation number: {self.param["product"]["ternary"]["oxidation_number"]}
    --------------------------------------------------------------------------""", end="")
            print(f"""
Quaternary Product: {self.param["active_element"]["secondary"]["elem"]} + {self.param["oxidant"]["secondary"]["elem"]}
    * Moles per cell: {self.param["product"]["quaternary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["product"]["quaternary"]["mass_per_cell"]} [kg]
    * Oxidation number: {self.param["product"]["quaternary"]["oxidation_number"]}
    --------------------------------------------------------------------------""", end="")

        elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
            print(f"""
Secondary Product: {self.param["active_element"]["secondary"]["elem"]} + {self.param["oxidant"]["primary"]["elem"]}
    * Moles per cell: {self.param["product"]["secondary"]["moles_per_cell"]} [mole]
    * Mass per cell: {self.param["product"]["secondary"]["mass_per_cell"]} [kg]
    * Secondary oxidation number: {self.param["product"]["secondary"]["oxidation_number"]}
    --------------------------------------------------------------------------""", end="")
