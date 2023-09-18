import numpy as np
import sys


class NucleationProbabilities:
    def __init__(self, param):
        self.nucl_prob_pp = np.full(param["n_cells_per_axis"], param["nucleation_probability"], dtype=float)
        self.nucl_prob_b = np.log(param["final_nucl_prob"] / param["nucleation_probability"])
        self.nucl_prob_a = param["nucleation_probability"]

        self.hf_init = param["het_factor"]
        self.hf_pp = np.full(param["n_cells_per_axis"], self.hf_init, dtype=float)
        self.hf_b = np.log(param["hf_deg_lim"])

        if param["max_neigh_numb"] == 0:
            if param["product"]["primary"]["oxidation_number"] > 1:
                self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"] *\
                                    param["product"]["primary"]["oxidation_number"] - 1
            else:
                self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"]
        else:
            self.n_neigh_init = param["max_neigh_numb"]

        self.n_neigh_pp = np.full(param["n_cells_per_axis"], self.n_neigh_init, dtype=float)
        self.oxidation_number = param["product"]["primary"]["oxidation_number"]
        self.n_neigh_a = self.n_neigh_init

        self.lowest_neigh_numb = param["lowest_neigh_numb"]
        self.n_neigh_b = np.log(self.lowest_neigh_numb / self.n_neigh_init)

        # self.const_b_pp = np.log(1 / (self.hf_init * self.nucl_prob_pp)) * (1 / (self.n_neigh_init - 1))
        # self.const_a_pp = (1 / (self.hf_init * self.nucl_prob_pp)) ** (-self.n_neigh_init / (self.n_neigh_init - 1))

        self.const_b_pp = np.log(self.hf_pp * self.nucl_prob_pp) / (self.oxidation_number - self.n_neigh_pp)
        self.const_a_pp = 1 / (np.e ** (self.const_b_pp * self.n_neigh_pp))

        self.prob_map = {0: 0, 1: 0.0049, 2: 0.006, 3: 0.27, 4: 0.293, 5: 0.42, 6: 1}

    def reset_constants(self, nucleation_probability, het_factor, hf_deg_lim):
        self.nucl_prob_pp[:] = nucleation_probability
        self.hf_pp[:] = het_factor
        self.hf_init = het_factor

        self.hf_b = np.log(hf_deg_lim)

        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def update_constants(self):
        self.const_b_pp = np.log(self.hf_pp * self.nucl_prob_pp) / (self.oxidation_number - self.n_neigh_pp)
        self.const_a_pp = 1 / (np.e ** (self.const_b_pp * self.n_neigh_pp))

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)

    def get_probabilities_from_map(self, numb_of_neighbours):
        rescaled = np.array(numb_of_neighbours / 4, dtype=int)
        return np.array([self.prob_map[key] for key in rescaled], dtype=float)

    def adapt_hf(self, page_ind, rel_phase_fraction):
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()

    def adapt_hf_n_neigh(self, page_ind, rel_phase_fraction):
        self.n_neigh_pp[page_ind] = self.n_neigh_a * np.e ** (self.n_neigh_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()

    def adapt_hf_n_neigh_nucl_prob(self, page_ind, rel_phase_fraction):
        self.n_neigh_pp[page_ind] = self.n_neigh_a * np.e ** (self.n_neigh_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.nucl_prob_pp[page_ind] = self.nucl_prob_a * np.e ** (self.nucl_prob_b * rel_phase_fraction)
        self.update_constants()

    def adapt_hf_nucl_prob(self, page_ind, rel_phase_fraction):
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.nucl_prob_pp[page_ind] = self.nucl_prob_a * np.e ** (self.nucl_prob_b * rel_phase_fraction)
        self.update_constants()

    def adapt_nucl_prob(self, page_ind, rel_phase_fraction):
        self.nucl_prob_pp[page_ind] = self.nucl_prob_a * np.e ** (self.nucl_prob_b * rel_phase_fraction)


class DissolutionProbabilities:
    def __init__(self, param):
        self.dissol_prob_pp = np.full(param["n_cells_per_axis"], param["dissolution_p"])
        self.dissol_prob_a = param["dissolution_p"]
        self.dissol_prob_b = np.log(param["final_dissol_prob"] / param["dissolution_p"])

        self.min_dissol_prob_pp = np.full(param["n_cells_per_axis"], param["min_dissol_prob"])
        self.min_dissol_prob_a = param["min_dissol_prob"]
        self.min_dissol_prob_b = np.log(param["final_min_dissol_prob"] / param["min_dissol_prob"])

        self.oxidation_number = param["product"]["primary"]["oxidation_number"]
        self.n_neigh_init = self.oxidation_number * 6

        self.case_a = self.n_neigh_init / (self.n_neigh_init - self.oxidation_number)
        self.case_b = self.oxidation_number / (self.n_neigh_init - self.oxidation_number)
        self.p3 = self.oxidation_number * 3

        self.hf_pp = np.full(param["n_cells_per_axis"], param["het_factor_dissolution"], dtype=float)
        self.hf_init = param["het_factor_dissolution"]
        self.hf_b = np.log(param["final_het_factor_dissol"] / param["het_factor_dissolution"])

        self.n_neigh_pp = np.full(param["n_cells_per_axis"], self.n_neigh_init, dtype=float)
        self.n_neigh_a = self.n_neigh_init
        self.lowest_neigh_numb = param["lowest_neigh_numb"]
        self.n_neigh_b = np.log(self.lowest_neigh_numb / self.n_neigh_init)

        self.bsf = param["block_scale_factor"]

        self.const_a_pp = (self.dissol_prob_pp ** self.case_a) / \
                          ((self.min_dissol_prob_pp ** self.case_b) * (self.hf_pp ** self.case_a))
        # self.const_b_pp = np.array(np.log((self.dissol_prob_pp ** (-self.case_b)) * (self.min_dissol_prob_pp ** self.case_b) *
        #                          (self.hf_pp ** self.case_b)), dtype=float)
        self.const_b_pp = np.array(np.log(self.dissol_prob_pp / (self.hf_pp * self.const_a_pp)) / self.oxidation_number,
                                   dtype=float)

    def reset_constants(self, nucleation_probability, het_factor, hf_deg_lim):
        self.dissol_prob_pp[:] = nucleation_probability
        self.hf_pp[:] = het_factor
        self.hf_init = het_factor

        self.hf_b = np.log(hf_deg_lim)

        self.const_a_pp = (1 / (self.hf_pp * self.dissol_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.dissol_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def update_constants(self):
        self.const_a_pp = (self.dissol_prob_pp ** self.case_a) / \
                          ((self.min_dissol_prob_pp ** self.case_b) * (self.hf_pp ** self.case_a))
        self.const_b_pp = np.array(np.log(self.dissol_prob_pp / (self.hf_pp * self.const_a_pp)) / self.oxidation_number,
                                   dtype=float)

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)

    def get_probabilities_block(self, numb_of_neighbours, page_ind):
        return (self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)) / self.bsf

    def adapt_hf(self, page_ind, rel_phase_fraction):
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob_pp[page_ind] = self.dissol_prob_a * np.e ** (self.dissol_prob_b * rel_phase_fraction)
        self.update_constants()

    def adapt_hf_n_neigh(self, page_ind, rel_phase_fraction):
        self.n_neigh_pp[page_ind] = self.n_neigh_a * np.e ** (self.n_neigh_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()

    def adapt_hf_n_neigh_dissol_prob(self, page_ind, rel_phase_fraction):
        self.n_neigh_pp[page_ind] = self.n_neigh_a * np.e ** (self.n_neigh_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.dissol_prob_pp[page_ind] = self.dissol_prob_a * np.e ** (self.dissol_prob_b * rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob_pp[page_ind] = self.dissol_prob_a * np.e ** (self.dissol_prob_b * rel_phase_fraction)
        self.min_dissol_prob_pp[page_ind] = self.min_dissol_prob_a * np.e ** (self.min_dissol_prob_b * rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob_hf(self, page_ind, rel_phase_fraction):
        self.dissol_prob_pp[page_ind] = self.dissol_prob_a * np.e ** (self.dissol_prob_b * rel_phase_fraction)
        self.min_dissol_prob_pp[page_ind] = self.min_dissol_prob_a * np.e ** (self.min_dissol_prob_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob_hf(self, page_ind, rel_phase_fraction):
        self.dissol_prob_pp[page_ind] = self.dissol_prob_a * np.e ** (self.dissol_prob_b * rel_phase_fraction)
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()


class NucleationProbabilitiesADJ:
    def __init__(self, param):
        self.nucl_prob = ExpFunct(param["n_cells_per_axis"], param["nucleation_probability"], param["final_nucl_prob"],
                                  -1, param["b_const_P0_nucl"])
        self.p1 = ExpFunct(param["n_cells_per_axis"], param["init_P1"], param["final_P1"], 1, param["b_const_P1"])

        if param["max_neigh_numb"] == 0:
            if param["product"]["primary"]["oxidation_number"] > 1:
                self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"] *\
                                    param["product"]["primary"]["oxidation_number"] - 1
            else:
                self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"]
        else:
            self.n_neigh_init = param["max_neigh_numb"]

        self.oxidation_number = param["product"]["primary"]["oxidation_number"]
        self.const_a_pp = np.full(param["n_cells_per_axis"], -1, dtype=float)

        self.b0 = -0.00001
        self.b1 = -0.5
        self.delt_b = self.b1 - self.b0
        self.const_b_pp = np.full(param["n_cells_per_axis"], self.b0, dtype=float)

        self.const_c_pp = np.log((1 - self.p1.values_pp) / (self.const_a_pp *
                                                    (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                     np.e ** (self.const_b_pp * self.oxidation_number))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.oxidation_number +
                                                                         self.const_c_pp)

        self.adapt_probabilities = None
        if param["nucl_adapt_function"] == 0:
            self.adapt_probabilities = self.adapt_nucl_prob
        elif param["nucl_adapt_function"] == 1:
            self.adapt_probabilities = self.adapt_p1
        elif param["nucl_adapt_function"] == 2:
            self.adapt_probabilities = self.adapt_p1_nucl_prob

    def update_constants(self):
        self.const_c_pp = np.log((1 - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.oxidation_number))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.oxidation_number +
                                                                         self.const_c_pp)

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                    self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]

    def adapt_nucl_prob(self, page_ind, gamma_primes):
        self.nucl_prob.update_values_at_pos(page_ind, gamma_primes)
        self.update_constants()

    def adapt_p1(self, page_ind, rel_phase_fraction):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.update_constants()

    def adapt_p1_nucl_prob(self, page_ind, rel_phase_fraction, gamma_primes):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.nucl_prob.update_values_at_pos(page_ind, gamma_primes)
        self.update_constants()


class DissolutionProbabilitiesADJ:
    def __init__(self, param):
        # self.dissol_prob_pp = np.full(param["n_cells_per_axis"], param["dissolution_p"])
        # self.dissol_prob_a = param["dissolution_p"]
        # self.dissol_prob_b = np.log(param["final_dissol_prob"] / param["dissolution_p"])

        self.dissol_prob = ExpFunct(param["n_cells_per_axis"], param["dissolution_p"], param["final_dissol_prob"],
                                    1, 50)

        # self.min_dissol_prob_pp = np.full(param["n_cells_per_axis"], param["min_dissol_prob"])
        # self.min_dissol_prob_a = param["min_dissol_prob"]
        # self.min_dissol_prob_b = np.log(param["final_min_dissol_prob"] / param["min_dissol_prob"])

        self.min_dissol_prob = ExpFunct(param["n_cells_per_axis"], param["min_dissol_prob"],
                                        param["final_min_dissol_prob"], -1, 50)

        # self.hf_pp = np.full(param["n_cells_per_axis"], param["het_factor_dissolution"], dtype=float)
        # self.hf_init = param["het_factor_dissolution"]
        # self.hf_b = np.log(param["final_het_factor_dissol"] / param["het_factor_dissolution"])

        self.p1 = ExpFunct(param["n_cells_per_axis"], param["init_P1_diss"], param["final_P1_diss"],
                           1, param["b_const_P1_diss"])

        self.oxidation_number = param["product"]["primary"]["oxidation_number"]
        self.n_neigh_init = self.oxidation_number * 6

        # self.case_a = self.n_neigh_init / (self.n_neigh_init - self.oxidation_number)
        # self.case_b = self.oxidation_number / (self.n_neigh_init - self.oxidation_number)
        self.p3 = self.oxidation_number * 3

        # self.n_neigh_pp = np.full(param["n_cells_per_axis"], self.n_neigh_init, dtype=float)
        # self.n_neigh_a = self.n_neigh_init
        # self.lowest_neigh_numb = param["lowest_neigh_numb"]
        # self.n_neigh_b = np.log(self.lowest_neigh_numb / self.n_neigh_init)

        self.bsf = param["block_scale_factor"]

        # self.const_a_pp = (self.dissol_prob.values_pp ** self.case_a) / \
        #                   ((self.min_dissol_prob.values_pp ** self.case_b) * (self.hf.values_pp ** self.case_a))
        # self.const_b_pp = np.array(np.log(self.dissol_prob.values_pp / (self.hf.values_pp * self.const_a_pp)) /
        #                            self.oxidation_number, dtype=float)

        self.const_a_pp = np.full(param["n_cells_per_axis"], 1, dtype=float)

        self.b0 = -0.00001
        self.b1 = -0.2
        self.delt_b = self.b1 - self.b0
        self.const_b_pp = np.full(param["n_cells_per_axis"], self.b0, dtype=float)

        self.const_c_pp = np.log((self.min_dissol_prob.values_pp - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.oxidation_number))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.oxidation_number +
                                                                         self.const_c_pp)

        self.adapt_probabilities = None
        if param["dissol_adapt_function"] == 0:
            self.adapt_probabilities = self.adapt_dissol_prob
        elif param["dissol_adapt_function"] == 1:
            self.adapt_probabilities = self.adapt_p1
        elif param["dissol_adapt_function"] == 2:
            self.adapt_probabilities = self.adapt_dissol_prob_min_dissol_prob
        elif param["dissol_adapt_function"] == 3:
            self.adapt_probabilities = self.adapt_dissol_prob_min_dissol_prob_p1
        elif param["dissol_adapt_function"] == 4:
            self.adapt_probabilities = self.adapt_dissol_prob_p1

    def update_constants(self):
        # self.const_a_pp = (self.dissol_prob.values_pp ** self.case_a) / \
        #                   ((self.min_dissol_prob.values_pp ** self.case_b) * (self.hf.values_pp ** self.case_a))
        # self.const_b_pp = np.array(np.log(self.dissol_prob.values_pp / (self.hf.values_pp * self.const_a_pp)) /
        #                            self.oxidation_number, dtype=float)
        self.const_c_pp = np.log((self.min_dissol_prob.values_pp - self.p1.values_pp) / (self.const_a_pp *
                                                            (np.e ** (self.const_b_pp * self.n_neigh_init) -
                                                             np.e ** (self.const_b_pp * self.oxidation_number))))
        self.const_d_pp = self.p1.values_pp - self.const_a_pp * np.e ** (self.const_b_pp * self.oxidation_number +
                                                                         self.const_c_pp)

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                    self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]

    def get_probabilities_block(self, numb_of_neighbours, page_ind):
        return (self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours +
                                                     self.const_c_pp[page_ind]) + self.const_d_pp[page_ind]) / self.bsf

    def adapt_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.update_constants()

    def adapt_p1(self, page_ind, rel_phase_fraction):
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.min_dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.update_constants()

    def adapt_dissol_prob_min_dissol_prob_p1(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.min_dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.update_constants()

    def adapt_dissol_prob_p1(self, page_ind, rel_phase_fraction):
        self.dissol_prob.update_values_at_pos(page_ind, rel_phase_fraction)
        self.p1.update_values_at_pos(page_ind, rel_phase_fraction)
        self.const_b_pp[page_ind] = self.delt_b * rel_phase_fraction + self.b0

        self.update_constants()


class ExpFunct:
    def __init__(self, length, y0, y1, const_a, const_b, x0=0, x1=1):
        """
        dy > 0, const_a = 1 and const_b > 0: _______/
        dy > 0, const_a = -1 and const_b < 0: /------

        dy < 0, const_a = 1 and const_b < 0: \ _______
        dy < 0, const_a = -1 and const_b > 0: ------\
        """
        dy = y1 - y0

        if (dy > 0 and const_a * const_b < 0) or (dy < 0 and const_a * const_b > 0):
            print("Wrong input into ExpFunct!!!")
            sys.exit()

        if dy != 0:
            const_c = np.log(dy/(const_a * (np.e**(const_b * x1) - np.e**(const_b * x0))))
            const_d = y0 - const_a * np.e ** (const_b * x0 + const_c)
        else:
            const_c = 0
            const_d = 0
            const_b = 0
            const_a = y0

        self.length = length
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.dy = dy
        self.const_a = const_a
        self.const_b = const_b
        self.const_c = const_c
        self.const_d = const_d

        self.values_pp = np.full(self.length, self.y0, dtype=float)

    def update_values_at_pos(self, positions, x):
        self.values_pp[positions] = self.const_a * np.e**(self.const_b * x + self.const_c) + self.const_d
