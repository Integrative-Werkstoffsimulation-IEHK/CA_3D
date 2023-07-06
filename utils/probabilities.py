import numpy as np


class NucleationProbabilities:
    def __init__(self, param):
        self.nucl_prob_pp = np.full(param["n_cells_per_axis"], param["nucleation_probability"])
        self.nucl_prob_b = np.log(param["final_nucl_prob"] / param["nucleation_probability"])
        self.nucl_prob_a = param["nucleation_probability"]

        self.hf_pp = np.full(param["n_cells_per_axis"], param["het_factor"], dtype=float)
        self.hf_init = param["het_factor"]
        self.hf_b = np.log(param["hf_deg_lim"])

        self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"] * param["product"]["primary"]["oxidation_number"] - 1
        self.n_neigh_pp = np.full(param["n_cells_per_axis"], self.n_neigh_init, dtype=float)
        self.n_neigh_a = self.n_neigh_init
        self.lowest_neigh_numb = param["lowest_neigh_numb"]
        self.n_neigh_b = np.log(self.lowest_neigh_numb/ self.n_neigh_init)

        self.const_a_pp = (1 / (self.hf_init * self.nucl_prob_pp)) ** (-self.n_neigh_init / (self.n_neigh_init - 1))
        self.const_b_pp = np.log(1 / (self.hf_init * self.nucl_prob_pp)) * (1 / (self.n_neigh_init - 1))

    def reset_constants(self, nucleation_probability, het_factor, hf_deg_lim):
        self.nucl_prob_pp[:] = nucleation_probability
        self.hf_pp[:] = het_factor
        self.hf_init = het_factor

        self.hf_b = np.log(hf_deg_lim)

        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def update_constants(self):
        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)

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


class DissolutionProbabilities:
    def __init__(self, param):
        self.nucl_prob_pp = np.full(param["n_cells_per_axis"], param["nucleation_probability"])
        self.nucl_prob_b = np.log(param["final_nucl_prob"] / param["nucleation_probability"])
        self.nucl_prob_a = param["nucleation_probability"]

        self.hf_pp = np.full(param["n_cells_per_axis"], param["het_factor"], dtype=float)
        self.hf_init = param["het_factor"]
        self.hf_b = np.log(param["hf_deg_lim"])

        self.n_neigh_init = param["product"]["primary"]["lind_flat_arr"] * param["product"]["primary"]["oxidation_number"] - 1
        self.n_neigh_pp = np.full(param["n_cells_per_axis"], self.n_neigh_init, dtype=float)
        self.n_neigh_a = self.n_neigh_init
        self.lowest_neigh_numb = param["lowest_neigh_numb"]
        self.n_neigh_b = np.log(self.lowest_neigh_numb/ self.n_neigh_init)

        self.const_a_pp = (1 / (self.hf_init * self.nucl_prob_pp)) ** (-self.n_neigh_init / (self.n_neigh_init - 1))
        self.const_b_pp = np.log(1 / (self.hf_init * self.nucl_prob_pp)) * (1 / (self.n_neigh_init - 1))

    def reset_constants(self, nucleation_probability, het_factor, hf_deg_lim):
        self.nucl_prob_pp[:] = nucleation_probability
        self.hf_pp[:] = het_factor
        self.hf_init = het_factor

        self.hf_b = np.log(hf_deg_lim)

        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def update_constants(self):
        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-self.n_neigh_pp / (self.n_neigh_pp - 1))
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / (self.n_neigh_pp - 1))

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)

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