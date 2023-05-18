import numpy as np
from math import *


class NucleationProbabilities:
    def __init__(self, param):
        self.const_e = 2.718281828
        self.nucleation_probability = np.full(param["n_cells_per_axis"], param["nucleation_probability"])
        self.het_factor = param["het_factor"]
        self.const_a = (1 / (self.het_factor * self.nucleation_probability[0])) ** (-6 / 5)
        self.const_b = log(1 / (self.het_factor * self.nucleation_probability[0])) * (1 / 5)

    def set_constants(self, nucleation_probability, het_factor):
        self.nucleation_probability[:] = nucleation_probability
        self.het_factor = het_factor
        self.const_a = (1 / (self.het_factor * self.nucleation_probability[0])) ** (-6 / 5)
        self.const_b = log(1 / (self.het_factor * self.nucleation_probability[0])) * (1 / 5)

    def update_constants(self):
        self.const_a = (1 / (self.het_factor * self.nucleation_probability[0])) ** (-6 / 5)
        self.const_b = log(1 / (self.het_factor * self.nucleation_probability[0])) * (1 / 5)

    def get_probabilities(self, numb_of_neighbours):
        return self.const_a * self.const_e ** (self.const_b * numb_of_neighbours)

