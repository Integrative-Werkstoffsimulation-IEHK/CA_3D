import numpy as np


class NucleationProbabilities:
    def __init__(self, param):
        self.nucl_prob_pp = np.full(param["n_cells_per_axis"], param["nucleation_probability"])
        self.hf_pp = np.full(param["n_cells_per_axis"], param["het_factor"], dtype=np.uint64)

        self.hf_init = param["het_factor"]
        self.hf_b = np.log(param["hf_deg_lim"])

        self.const_a_pp = (1 / (self.hf_init * self.nucl_prob_pp)) ** (-6 / 5)
        self.const_b_pp = np.log(1 / (self.hf_init * self.nucl_prob_pp)) * (1 / 5)

    def reset_constants(self, nucleation_probability, het_factor, hf_deg_lim):
        self.nucl_prob_pp[:] = nucleation_probability
        self.hf_pp[:] = het_factor
        self.hf_init = het_factor

        self.hf_b = np.log(hf_deg_lim)

        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-6 / 5)
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / 5)

    def update_constants(self):
        self.const_a_pp = (1 / (self.hf_pp * self.nucl_prob_pp)) ** (-6 / 5)
        self.const_b_pp = np.log(1 / (self.hf_pp * self.nucl_prob_pp)) * (1 / 5)

    def get_probabilities(self, numb_of_neighbours, page_ind):
        return self.const_a_pp[page_ind] * np.e ** (self.const_b_pp[page_ind] * numb_of_neighbours)

    def adapt_hf(self, page_ind, rel_phase_fraction):
        self.hf_pp[page_ind] = self.hf_init * np.e ** (self.hf_b * rel_phase_fraction)
        self.update_constants()
