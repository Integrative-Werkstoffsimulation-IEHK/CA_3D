import numpy as np
from math import *
import random
from utilities import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import progressbar


class Cell:
    def __init__(self, element, mass, shape, diff_coeff, utils):
        self.utils = utils
        self.element = element
        self.mass = mass
        self.diff_coeff = diff_coeff
        self.shape = shape
        self.position = np.array([np.random.randint(self.shape[0]), np.random.randint(self.shape[0]), 0], dtype=int)
        self.position_flat = np.ravel_multi_index(self.position, self.shape)
        self.probs = self.utils.calc_prob(self.diff_coeff)
        self.p = self.probs["p"]
        self.p3 = self.probs["p3"]
        self.p0 = self.probs["p0"]
        self.probabilities = np.array([self.p0, self.p, self.p, self.p3, self.p, self.p])
        self.bal_direction = np.array([0, 0, 1], dtype=int)
        self.mul_roll = np.array([[1, 0], [1, 1], [1, 2], [-1, 0], [-1, 1], [-1, 2]], dtype=int)
        self.in_scope = True

    def switch_direction(self):
        index = np.random.choice([0, 1, 2, 3, 4, 5], p=self.probabilities)
        self.bal_direction = np.roll(self.bal_direction, -1 * self.mul_roll[index][1]) * self.mul_roll[index][0]

    def move(self):
        self.position += self.bal_direction
        # periodic boundary condition
        if self.position[0] > self.shape[0] - 1:
            self.position[0] = 0
        if self.position[1] > self.shape[0] - 1:
            self.position[1] = 0
        if self.position[2] > self.shape[0] - 1:
            self.position[2] = 0
            self.in_scope = False

        if self.position[0] < 0:
            self.position[0] = self.shape[0] - 1
        if self.position[1] < 0:
            self.position[1] = self.shape[0] - 1
        if self.position[2] < 0:
            self.position[2] = self.shape[0] - 1
            self.in_scope = False

        self.position_flat = np.ravel_multi_index(self.position, self.shape)

    def update_probabilities(self, new_diff_coeff):
        self.p, self.p3, self.p0 = self.utils.calc_prob(new_diff_coeff)
        self.probabilities = np.array([self.p0, self.p, self.p, self.p3, self.p, self.p])


if __name__ == "__main__":
    def plot_3d(cells_to_plot):
        x = [cell.position[2] for cell in cells_to_plot]
        y = [cell.position[1] for cell in cells_to_plot]
        z = [cell.position[0] for cell in cells_to_plot]

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(x, y, z, marker=',', color='b', s=1)
        ax1.set_xlim3d(0, 100)
        ax1.set_ylim3d(0, 100)
        ax1.set_zlim3d(0, 100)
        plt.show()


    param = {"matrix_elem": "Ni",
             "cond_inward": "Test",  # [m^2/sek]
             "cond_outward": "Test_slower",  # [m^2/sek]
             "diff_in_precipitation": 6.18 * 10 ** -14,  # [m^2/sek]
             "diff_out_precipitation": 6.18 * 10 ** -16,  # [m^2/sek]
             "temperature": 1100,  # °C
             "n_cells_per_axis": 102,  # ONLY MULTIPLES OF 3 ARE ALLOWED!!!!!!!
             "n_iterations": 100,  # must be >= n_cells_per_axis!!!!!!!!!!!
             "sim_time": 7200,  # [sek]
             "size": 0.001,  # [m]

             "count_only_cells": True,
             "diff_elem_conc": 0.000005,  # mass/cells fraction (depends on the "count_only_cells" flag!)
             "active_elem_conc": 0.02,  # cells fraction
             "active_elem_conc_real": 0.02,  # mass fraction
             "threshold_inward": 2,
             "threshold_outward": 2,
             "threshold_growth_inward": 1,
             "threshold_growth_outward": 1,
             "sol_prod": 1.4 * 10 ** -6,

             "inward_diffusion": True,
             "outward_diffusion": False,
             "compute_precipitations": True,
             "grow_precipitations": False,
             "consider_diffusion_in_precipitation": False,
             "save_whole": False,
             "save_precipitation_front": True
             }

    util = Utils(param)
    cells = np.array([Cell("Ni", 5, (100, 100, 100), 6.18 * 10 ** -12, util)])

    for iteration in progressbar.progressbar(range(100)):
        new_cells = np.array([Cell("Ni", 5, (100, 100, 100), 6.18 * 10 ** -12, util) for i in range(100)])
        cells = np.concatenate((cells, new_cells))

        to_del = np.where(np.logical_not([cell.in_scope for cell in cells]))[0]
        cells = np.delete(cells, to_del)

        [cell.switch_direction() for cell in cells]
        [cell.move() for cell in cells]

    plot_3d(cells)

    # def animate_3d():
    #     cells = np.array([Cell("Ni", 5, (100, 100, 100), 6.18 * 10 ** -12, util)])
    #     def animate(iteration):
    #         ax1.cla()
    #         # new_cells = np.array([Cell("Ni", 5, (100, 100, 100), 6.18 * 10 ** -16, util) for i in range(100)])
    #         # cells = np.concatenate((cells, new_cells))
    #         # [cell.switch_direction() for cell in cells]
    #         # [cell.move() for cell in cells]
    #         # ax1.scatter(cell.position[2], cell.position[1], cell.position[0], marker=',', color='b', s=10)
    #         # ax1.set_xlim3d(0, 100)
    #         # ax1.set_ylim3d(0, 100)
    #         # ax1.set_zlim3d(0, 100)
    #
    #         new_cells = np.array([Cell("Ni", 5, (100, 100, 100), 6.18 * 10 ** -12, util) for i in range(100)])
    #         cells = np.concatenate((cells, new_cells))
    #
    #         to_del = np.where(np.logical_not([cell.in_scope for cell in cells]))[0]
    #         cells = np.delete(cells, to_del)
    #
    #         [cell.switch_direction() for cell in cells]
    #         [cell.move() for cell in cells]
    #
    #         ax1.scatter(cells.position[2], cells.position[1], cells.position[0], marker=',', color='b', s=10)
    #         ax1.set_xlim3d(0, 100)
    #         ax1.set_ylim3d(0, 100)
    #         ax1.set_zlim3d(0, 100)
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     animation = FuncAnimation(fig, animate)
    #     plt.show()
    #
    #
    # animate_3d()