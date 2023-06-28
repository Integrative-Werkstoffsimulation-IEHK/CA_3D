import matplotlib.pyplot as plt
import sqlite3 as sql
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy import special
from math import *
import numpy as np
from utils import utilities, physical_data, templates


class Visualisation:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sql.connect(self.db_name)
        self.c = self.conn.cursor()
        self.param = None
        self.axlim = None
        self.shape = None
        self.last_i = None
        self.generate_param_from_db()
        self.cell_size = 5
        self.linewidth = 1
        self.cm = {1: np.array([255, 200, 200])/255.0,
                   2: np.array([255, 75, 75])/255.0,
                   3: np.array([220, 0, 0])/255.0,
                   4: np.array([120, 0, 0])/255.0}

    def generate_param_from_db(self):
        user_input = templates.DEFAULT_PARAM
        self.c.execute("SELECT * from user_input")
        user_input_from_db = self.c.fetchall()[0]
        for position, key in enumerate(user_input):
            if 2 < position < len(user_input_from_db) + 3:
                user_input[key] = user_input_from_db[position - 3]

        self.c.execute("SELECT * from element_0")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["active_element"]["primary"]["elem"] = elem_data_from_db[0]
        user_input["active_element"]["primary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["active_element"]["primary"]["mass_concentration"] = elem_data_from_db[2]
        user_input["active_element"]["primary"]["cells_concentration"] = elem_data_from_db[3]
        self.c.execute("SELECT * from element_1")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["active_element"]["secondary"]["elem"] = elem_data_from_db[0]
        user_input["active_element"]["secondary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["active_element"]["secondary"]["mass_concentration"] = elem_data_from_db[2]
        user_input["active_element"]["secondary"]["cells_concentration"] = elem_data_from_db[3]
        self.c.execute("SELECT * from element_2")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["oxidant"]["primary"]["elem"] = elem_data_from_db[0]
        user_input["oxidant"]["primary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["oxidant"]["primary"]["cells_concentration"] = elem_data_from_db[2]
        self.c.execute("SELECT * from element_3")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["oxidant"]["secondary"]["elem"] = elem_data_from_db[0]
        user_input["oxidant"]["secondary"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["oxidant"]["secondary"]["cells_concentration"] = elem_data_from_db[2]
        self.c.execute("SELECT * from element_4")
        elem_data_from_db = self.c.fetchall()[0]
        user_input["matrix_elem"]["elem"] = elem_data_from_db[0]
        user_input["matrix_elem"]["diffusion_condition"] = elem_data_from_db[1]
        user_input["matrix_elem"]["concentration"] = elem_data_from_db[2]
        utils = utilities.Utils(user_input)
        utils.generate_param()
        utils.print_init_var()
        self.c.execute("SELECT last_i from time_parameters")
        self.last_i = self.c.fetchone()[0]
        self.compute_elapsed_time()
        self.param = utils.param
        self.axlim = self.param["n_cells_per_axis"]
        self.shape = (self.axlim, self.axlim, self.axlim)

        if not self.param["inward_diffusion"]:
            print("No INWARD data!")
        if not self.param["compute_precipitations"]:
            print("No PRECIPITATION data!")
        if not self.param["outward_diffusion"]:
            print("No OUTWARD data!")

    def compute_elapsed_time(self):
        self.c.execute("SELECT elapsed_time from time_parameters")
        elapsed_time_sek = np.array(self.c.fetchall()[0])
        if elapsed_time_sek != 0:
            h = elapsed_time_sek // 3600
            m = (elapsed_time_sek - h * 3600) // 60
            s = elapsed_time_sek - h * 3600 - m * 60
            message = f'{int(h)}h:{int(m)}m:{int(s)}s'
        else:
            message = f'Simulation was interrupted at iteration = {self.last_i}'

        print(f"""
TIME:------------------------------------------------------------
ELAPSED TIME: {message}
-----------------------------------------------------------------""")

    def animate_3d(self, animate_separate=False, const_cam_pos=False):
        if self.param["save_whole"]:
            def animate_sep(iteration):
                ax_inward.cla()
                ax_sinward.cla()
                ax_outward.cla()
                ax_soutward.cla()
                ax_precip.cla()
                ax_sprecip.cla()
                ax_tprecip.cla()
                ax_qtprecip.cla()
                if self.param["inward_diffusion"]:
                    self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_inward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
                                          s=self.cell_size * (72. / fig.dpi) ** 2)
                    if self.param["secondary_oxidant_exists"]:
                        self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_sinward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["outward_diffusion"]:
                    self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_outward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g', s=3)
                    if self.param["secondary_active_element_exists"]:
                        self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_soutward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                                s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["compute_precipitations"]:
                    self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_precip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                                          s=self.cell_size * (72. / fig.dpi) ** 2)

                    if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                        self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)

                        self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_tprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)

                        self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_qtprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                                s=self.cell_size * (72. / fig.dpi) ** 2)

                    elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                        self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                ax_inward.set_xlim3d(0, self.axlim)
                ax_inward.set_ylim3d(0, self.axlim)
                ax_inward.set_zlim3d(0, self.axlim)
                ax_sinward.set_xlim3d(0, self.axlim)
                ax_sinward.set_ylim3d(0, self.axlim)
                ax_sinward.set_zlim3d(0, self.axlim)
                ax_outward.set_xlim3d(0, self.axlim)
                ax_outward.set_ylim3d(0, self.axlim)
                ax_outward.set_zlim3d(0, self.axlim)
                ax_soutward.set_xlim3d(0, self.axlim)
                ax_soutward.set_ylim3d(0, self.axlim)
                ax_soutward.set_zlim3d(0, self.axlim)
                ax_precip.set_xlim3d(0, self.axlim)
                ax_precip.set_ylim3d(0, self.axlim)
                ax_precip.set_zlim3d(0, self.axlim)
                ax_sprecip.set_xlim3d(0, self.axlim)
                ax_sprecip.set_ylim3d(0, self.axlim)
                ax_sprecip.set_zlim3d(0, self.axlim)
                ax_tprecip.set_xlim3d(0, self.axlim)
                ax_tprecip.set_ylim3d(0, self.axlim)
                ax_tprecip.set_zlim3d(0, self.axlim)
                ax_qtprecip.set_xlim3d(0, self.axlim)
                ax_qtprecip.set_ylim3d(0, self.axlim)
                ax_qtprecip.set_zlim3d(0, self.axlim)
                if const_cam_pos:
                    azim = -70
                    elev = 30
                    dist = 8
                    ax_inward.azim = azim
                    ax_inward.elev = elev
                    ax_inward.dist = dist
                    ax_sinward.azim = azim
                    ax_sinward.elev = elev
                    ax_sinward.dist = dist
                    ax_outward.azim = azim
                    ax_outward.elev = elev
                    ax_outward.dist = dist
                    ax_soutward.azim = azim
                    ax_soutward.elev = elev
                    ax_soutward.dist = dist
                    ax_sprecip.azim = azim
                    ax_sprecip.elev = elev
                    ax_sprecip.dist = dist
                    ax_sprecip.azim = azim
                    ax_sprecip.elev = elev
                    ax_sprecip.dist = dist

            def animate(iteration):
                ax_all.cla()
                # ax_all.dist = 4
                if self.param["inward_diffusion"]:
                    self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                #     if self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                # if self.param["outward_diffusion"]:
                #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                #     items = np.array(self.c.fetchall())
                #     if np.any(items):
                #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
                #                        s=self.cell_size * (72. / fig.dpi) ** 2)
                #     if self.param["secondary_active_element_exists"]:
                #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                # if self.param["compute_precipitations"]:
                #     self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                #     items = np.array(self.c.fetchall())
                #     if np.any(items):
                #         # items = items.transpose()
                #         # data = np.zeros(self.shape, dtype=bool)
                #         # data[items[0], items[1], items[2]] = True
                #         # ax_all.voxels(data, facecolors="r")
                #         # plt.savefig(f'W:/SIMCA/test_runs_data/{iteration}.jpeg')
                #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                #                        s=self.cell_size * (72. / fig.dpi) ** 2)
                #
                #     if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #
                #         self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #
                #         self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #
                #     elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)

                ax_all.set_xlim3d(0, self.axlim)
                ax_all.set_ylim3d(0, self.axlim)
                ax_all.set_zlim3d(0, self.axlim)

                if const_cam_pos:
                    ax_all.azim = -45
                    ax_all.elev = 22
                    ax_all.dist = 9

            fig = plt.figure()
            # fig.set_size_inches(18.5, 10.5)
            if animate_separate:
                ax_inward = fig.add_subplot(341, projection='3d')
                ax_sinward = fig.add_subplot(345, projection='3d')
                ax_outward = fig.add_subplot(342, projection='3d')
                ax_soutward = fig.add_subplot(346, projection='3d')

                ax_precip = fig.add_subplot(349, projection='3d')
                ax_sprecip = fig.add_subplot(3, 4, 10, projection='3d')
                ax_tprecip = fig.add_subplot(3, 4, 11, projection='3d')
                ax_qtprecip = fig.add_subplot(3, 4, 12, projection='3d')
                animation = FuncAnimation(fig, animate_sep)

            else:
                ax_all = fig.add_subplot(111, projection='3d')
                animation = FuncAnimation(fig, animate)
            plt.show()
            # plt.savefig(f'W:/SIMCA/test_runs_data/{"_"}.jpeg')
        else:
            return print("No Data To Animate!")

    def plot_3d(self, plot_separate=False, iteration=None, const_cam_pos=False):
        if iteration is None:
            iteration = self.last_i
        fig = plt.figure()
        if plot_separate:
            ax_inward = fig.add_subplot(341, projection='3d')
            ax_sinward = fig.add_subplot(345, projection='3d')
            ax_outward = fig.add_subplot(342, projection='3d')
            ax_soutward = fig.add_subplot(346, projection='3d')

            ax_precip = fig.add_subplot(349, projection='3d')
            ax_sprecip = fig.add_subplot(3,4,10, projection='3d')
            ax_tprecip = fig.add_subplot(3,4,11, projection='3d')
            ax_qtprecip = fig.add_subplot(3,4,12, projection='3d')

            if self.param["inward_diffusion"]:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_inward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sinward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["outward_diffusion"]:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_outward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_active_element_exists"]:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_soutward.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["compute_precipitations"]:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ax_precip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)

                if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_tprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkgreen',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_qtprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)

                elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_sprecip.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

            ax_inward.set_xlim3d(0, self.axlim)
            ax_inward.set_ylim3d(0, self.axlim)
            ax_inward.set_zlim3d(0, self.axlim)
            ax_sinward.set_xlim3d(0, self.axlim)
            ax_sinward.set_ylim3d(0, self.axlim)
            ax_sinward.set_zlim3d(0, self.axlim)
            ax_outward.set_xlim3d(0, self.axlim)
            ax_outward.set_ylim3d(0, self.axlim)
            ax_outward.set_zlim3d(0, self.axlim)
            ax_soutward.set_xlim3d(0, self.axlim)
            ax_soutward.set_ylim3d(0, self.axlim)
            ax_soutward.set_zlim3d(0, self.axlim)
            ax_precip.set_xlim3d(0, self.axlim)
            ax_precip.set_ylim3d(0, self.axlim)
            ax_precip.set_zlim3d(0, self.axlim)
            ax_sprecip.set_xlim3d(0, self.axlim)
            ax_sprecip.set_ylim3d(0, self.axlim)
            ax_sprecip.set_zlim3d(0, self.axlim)
            ax_tprecip.set_xlim3d(0, self.axlim)
            ax_tprecip.set_ylim3d(0, self.axlim)
            ax_tprecip.set_zlim3d(0, self.axlim)
            ax_qtprecip.set_xlim3d(0, self.axlim)
            ax_qtprecip.set_ylim3d(0, self.axlim)
            ax_qtprecip.set_zlim3d(0, self.axlim)

            if const_cam_pos:
                azim = -92
                elev = 0
                dist = 8
                ax_inward.azim = azim
                ax_inward.elev = elev
                ax_inward.dist = dist
                ax_sinward.azim = azim
                ax_sinward.elev = elev
                ax_sinward.dist = dist
                ax_outward.azim = azim
                ax_outward.elev = elev
                ax_outward.dist = dist
                ax_soutward.azim = azim
                ax_soutward.elev = elev
                ax_soutward.dist = dist
                ax_precip.azim = azim
                ax_precip.elev = elev
                ax_precip.dist = dist
                ax_sprecip.azim = azim
                ax_sprecip.elev = elev
                ax_sprecip.dist = dist
                ax_tprecip.azim = azim
                ax_tprecip.elev = elev
                ax_tprecip.dist = dist
                ax_qtprecip.azim = azim
                ax_qtprecip.elev = elev
                ax_qtprecip.dist = dist
        else:
            ax_all = fig.add_subplot(111, projection='3d')
            # if self.param["inward_diffusion"]:
            #     self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='b',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #     if self.param["secondary_oxidant_exists"]:
            #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='deeppink',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            # if self.param["outward_diffusion"]:
            #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='g',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #     if self.param["secondary_active_element_exists"]:
            #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkorange',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["compute_precipitations"]:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    counts = np.unique(np.ravel_multi_index(items.transpose(), self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
                    counts = np.array(counts[1], dtype=np.ubyte)

                    # for grade in range(1, 5):
                    #     grade_ind = np.where(counts == grade)[0]
                    #     ax_all.scatter(dec[grade_ind, 2], dec[grade_ind, 1], dec[grade_ind, 0], marker=',',
                    #                    color=self.cm[grade], s=self.cell_size * (72. / fig.dpi) ** 2)

                    full_ind = np.where(counts == 4)[0]

                    fulls = dec[full_ind]
                    not_fulls = np.delete(dec, full_ind, axis=0)

                    ax_all.scatter(fulls[:, 2], fulls[:, 1], fulls[:, 0], marker=',', color="darkred",
                                   s=self.cell_size * (72. / fig.dpi) ** 2)

                    ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], not_fulls[:, 0], marker=',', color='r',
                                   s=self.cell_size * (72. / fig.dpi) ** 2)

                if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='darkgreen',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

                elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ax_all.scatter(items[:, 2], items[:, 1], items[:, 0], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)

            ax_all.set_xlim3d(0, self.axlim)
            ax_all.set_ylim3d(0, self.axlim)
            ax_all.set_zlim3d(0, self.axlim)
            if const_cam_pos:
                ax_all.azim = -57
                ax_all.elev = 30
                ax_all.dist = 9
        # self.conn.commit()
        # plt.savefig(f'W:/SIMCA/test_runs_data/{iteration}.jpeg')
        plt.show()

    def plot_2d(self, plot_separate=False, iteration=None, slice_pos=None):
        if iteration is None:
            iteration = self.last_i
        if slice_pos is None:
            slice_pos = int(self.param["n_cells_per_axis"] / 2)
        fig = plt.figure()
        if plot_separate:
            ax_inward = fig.add_subplot(341)
            ax_sinward = fig.add_subplot(345)
            ax_outward = fig.add_subplot(342)
            ax_soutward = fig.add_subplot(346)

            ax_precip = fig.add_subplot(349)
            ax_sprecip = fig.add_subplot(3, 4, 10)
            ax_tprecip = fig.add_subplot(3, 4, 11)
            ax_qtprecip = fig.add_subplot(3, 4, 12)

            if self.param["inward_diffusion"]:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_inward.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sinward.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["outward_diffusion"]:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_outward.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_active_element_exists"]:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_soutward.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["compute_precipitations"]:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)
                    ax_precip.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                                      s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_tprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_qtprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                            s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)

            ax_inward.set_xlim(0, self.axlim)
            ax_inward.set_ylim(0, self.axlim)
            ax_sinward.set_xlim(0, self.axlim)
            ax_sinward.set_ylim(0, self.axlim)
            ax_outward.set_xlim(0, self.axlim)
            ax_outward.set_ylim(0, self.axlim)
            ax_soutward.set_xlim(0, self.axlim)
            ax_soutward.set_ylim(0, self.axlim)
            ax_precip.set_xlim(0, self.axlim)
            ax_precip.set_ylim(0, self.axlim)
            ax_sprecip.set_xlim(0, self.axlim)
            ax_sprecip.set_ylim(0, self.axlim)
            ax_tprecip.set_xlim(0, self.axlim)
            ax_tprecip.set_ylim(0, self.axlim)
            ax_qtprecip.set_xlim(0, self.axlim)
            ax_qtprecip.set_ylim(0, self.axlim)
        else:
            ax_all = fig.add_subplot(111)
            # if self.param["inward_diffusion"]:
            #     self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ind = np.where(items[:, 0] == slice_pos)
            #         ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #     if self.param["secondary_oxidant_exists"]:
            #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ind = np.where(items[:, 0] == slice_pos)
            #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            # if self.param["outward_diffusion"]:
            #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            #     items = np.array(self.c.fetchall())
            #     if np.any(items):
            #         ind = np.where(items[:, 0] == slice_pos)
            #         ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
            #                        s=self.cell_size * (72. / fig.dpi) ** 2)
            #     if self.param["secondary_active_element_exists"]:
            #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
            #         items = np.array(self.c.fetchall())
            #         if np.any(items):
            #             ind = np.where(items[:, 0] == slice_pos)
            #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
            #                            s=self.cell_size * (72. / fig.dpi) ** 2)
            if self.param["compute_precipitations"]:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    ind = np.where(items[:, 0] == slice_pos)

                    items = np.array(items[ind]).transpose()

                    counts = np.unique(np.ravel_multi_index(items, self.shape), return_counts=True)
                    dec = np.array(np.unravel_index(counts[0], self.shape), dtype=np.short).transpose()
                    counts = np.array(counts[1], dtype=np.ubyte)

                    for grade in range(1, 5):
                        grade_ind = np.where(counts == grade)[0]
                        ax_all.scatter(dec[grade_ind, 2], dec[grade_ind, 1], marker=',',
                                       color=self.cm[grade], s=self.cell_size * (72. / fig.dpi) ** 2)

                    # full_ind = np.where(counts == 4)[0]
                    #
                    # fulls = dec[full_ind]
                    # not_fulls = np.delete(dec, full_ind, axis=0)
                    #
                    # ax_all.scatter(fulls[:, 2], fulls[:, 1], marker=',', color='darkred',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2)
                    #
                    # ax_all.scatter(not_fulls[:, 2], not_fulls[:, 1], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2)

                    # ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                    #                s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
            ax_all.set_xlim(0, self.axlim)
            ax_all.set_ylim(0, self.axlim)
        self.conn.commit()
        # plt.savefig(f'{iteration}.jpeg')
        plt.show()

    def animate_2d(self, plot_separate=False, slice_pos=None):
        if self.param["save_whole"]:
            def animate_sep(iteration):
                if self.param["inward_diffusion"]:
                    ax_inward.cla()
                    self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_inward.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                          s=self.cell_size * (72. / fig.dpi) ** 2)
                    if self.param["secondary_oxidant_exists"]:
                        ax_sinward.cla()
                        self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_sinward.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["outward_diffusion"]:
                    ax_outward.cla()
                    self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_outward.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                                           s=self.cell_size * (72. / fig.dpi) ** 2)
                    if self.param["secondary_active_element_exists"]:
                        ax_soutward.cla()
                        self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_soutward.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                                                s=self.cell_size * (72. / fig.dpi) ** 2)
                if self.param["compute_precipitations"]:
                    self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_precip.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                                          s=self.cell_size * (72. / fig.dpi) ** 2)
                    if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                        self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                        self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_tprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                        self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_qtprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                                                s=self.cell_size * (72. / fig.dpi) ** 2)
                    elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                        self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                        items = np.array(self.c.fetchall())
                        if np.any(items):
                            ind = np.where(items[:, 0] == slice_pos)
                            ax_sprecip.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                                               s=self.cell_size * (72. / fig.dpi) ** 2)
                ax_inward.set_xlim(0, self.axlim)
                ax_inward.set_ylim(0, self.axlim)
                ax_sinward.set_xlim(0, self.axlim)
                ax_sinward.set_ylim(0, self.axlim)
                ax_outward.set_xlim(0, self.axlim)
                ax_outward.set_ylim(0, self.axlim)
                ax_soutward.set_xlim(0, self.axlim)
                ax_soutward.set_ylim(0, self.axlim)
                ax_precip.set_xlim(0, self.axlim)
                ax_precip.set_ylim(0, self.axlim)
                ax_sprecip.set_xlim(0, self.axlim)
                ax_sprecip.set_ylim(0, self.axlim)

            def animate(iteration):
                ax_all.cla()
                if self.param["inward_diffusion"]:
                    self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        ind = np.where(items[:, 0] == slice_pos)
                        ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='b',
                                       s=self.cell_size * (72. / fig.dpi) ** 2)
                #     if self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='deeppink',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                # if self.param["outward_diffusion"]:
                #     self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                #     items = np.array(self.c.fetchall())
                #     if np.any(items):
                #         ind = np.where(items[:, 0] == slice_pos)
                #         ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='g',
                #                        s=self.cell_size * (72. / fig.dpi) ** 2)
                #     if self.param["secondary_active_element_exists"]:
                #         self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkorange',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                # if self.param["compute_precipitations"]:
                #     self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                #     items = np.array(self.c.fetchall())
                #     if np.any(items):
                #         ind = np.where(items[:, 0] == slice_pos)
                #         ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='r',
                #                        s=self.cell_size * (72. / fig.dpi) ** 2)
                #     if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #         self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='darkgreen',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #         self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='steelblue',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                #     elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                #         self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                #         items = np.array(self.c.fetchall())
                #         if np.any(items):
                #             ind = np.where(items[:, 0] == slice_pos)
                #             ax_all.scatter(items[ind, 2], items[ind, 1], marker=',', color='cyan',
                #                            s=self.cell_size * (72. / fig.dpi) ** 2)
                ax_all.set_xlim(0, self.axlim)
                ax_all.set_ylim(0, self.axlim)

            if slice_pos is None:
                slice_pos = int(self.param["n_cells_per_axis"] / 2)
            fig = plt.figure()
            if plot_separate:
                ax_inward = fig.add_subplot(341)
                ax_sinward = fig.add_subplot(345)
                ax_outward = fig.add_subplot(342)
                ax_soutward = fig.add_subplot(346)

                ax_precip = fig.add_subplot(349)
                ax_sprecip = fig.add_subplot(3, 4, 10)
                ax_tprecip = fig.add_subplot(3, 4, 11)
                ax_qtprecip = fig.add_subplot(3, 4, 12)
                animation = FuncAnimation(fig, animate_sep)
            else:
                ax_all = fig.add_subplot(111)
                animation = FuncAnimation(fig, animate)
            plt.show()
        else:
            return print("No Data To Animate!")

    def animate_concentration(self, analytic_sol=False, conc_type="atomic"):
        def animate(iteration):
            inward = np.zeros(self.axlim, dtype=int)
            inward_moles = np.zeros(self.axlim, dtype=int)
            inward_mass = np.zeros(self.axlim, dtype=int)

            sinward = np.zeros(self.axlim, dtype=int)
            sinward_moles = np.zeros(self.axlim, dtype=int)
            sinward_mass = np.zeros(self.axlim, dtype=int)

            outward = np.zeros(self.axlim, dtype=int)
            outward_moles = np.zeros(self.axlim, dtype=int)
            outward_mass = np.zeros(self.axlim, dtype=int)
            outward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            soutward = np.zeros(self.axlim, dtype=int)
            soutward_moles = np.zeros(self.axlim, dtype=int)
            soutward_mass = np.zeros(self.axlim, dtype=int)
            soutward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            primary_product = np.zeros(self.axlim, dtype=int)
            primary_product_moles = np.zeros(self.axlim, dtype=int)
            primary_product_mass = np.zeros(self.axlim, dtype=int)
            primary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            secondary_product = np.zeros(self.axlim, dtype=int)
            secondary_product_moles = np.zeros(self.axlim, dtype=int)
            secondary_product_mass = np.zeros(self.axlim, dtype=int)
            secondary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            ternary_product = np.zeros(self.axlim, dtype=int)
            ternary_product_moles = np.zeros(self.axlim, dtype=int)
            ternary_product_mass = np.zeros(self.axlim, dtype=int)
            ternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            quaternary_product = np.zeros(self.axlim, dtype=int)
            quaternary_product_moles = np.zeros(self.axlim, dtype=int)
            quaternary_product_mass = np.zeros(self.axlim, dtype=int)
            quaternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

            if self.param["inward_diffusion"]:
                self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                inward_moles = inward * self.param["oxidant"]["primary"]["moles_per_cell"]
                inward_mass = inward * self.param["oxidant"]["primary"]["mass_per_cell"]

                if self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    sinward_moles = sinward * self.param["oxidant"]["secondary"]["moles_per_cell"]
                    sinward_mass = sinward * self.param["oxidant"]["secondary"]["mass_per_cell"]

            if self.param["outward_diffusion"]:
                self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                outward_moles = outward * self.param["active_element"]["primary"]["moles_per_cell"]
                outward_mass = outward * self.param["active_element"]["primary"]["mass_per_cell"]
                outward_eq_mat_moles = outward * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
                if self.param["secondary_active_element_exists"]:
                    self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    soutward_moles = soutward * self.param["active_element"]["secondary"]["moles_per_cell"]
                    soutward_mass = soutward * self.param["active_element"]["secondary"]["mass_per_cell"]
                    soutward_eq_mat_moles = soutward * self.param["active_element"]["secondary"][
                        "eq_matrix_moles_per_cell"]

            if self.param["compute_precipitations"]:
                self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    primary_product_moles = primary_product * self.param["product"]["primary"]["moles_per_cell"]
                    primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]
                    primary_product_eq_mat_moles = primary_product * self.param["active_element"]["primary"][
                        "eq_matrix_moles_per_cell"]

                if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.param["product"]["secondary"]["moles_per_cell"]
                    secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]
                    secondary_product_eq_mat_moles = secondary_product * self.param["active_element"]["secondary"][
                        "eq_matrix_moles_per_cell"]

                    self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    ternary_product_moles = ternary_product * self.param["product"]["ternary"]["moles_per_cell"]
                    ternary_product_mass = ternary_product * self.param["product"]["ternary"]["mass_per_cell"]
                    ternary_product_eq_mat_moles = ternary_product * self.param["active_element"]["primary"][
                        "eq_matrix_moles_per_cell"]

                    self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    quaternary_product_moles = quaternary_product * self.param["product"]["quaternary"][
                        "moles_per_cell"]
                    quaternary_product_mass = quaternary_product * self.param["product"]["quaternary"]["mass_per_cell"]
                    quaternary_product_eq_mat_moles = quaternary_product * self.param["active_element"]["secondary"][
                        "eq_matrix_moles_per_cell"]

                elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                    self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                    items = np.array(self.c.fetchall())
                    if np.any(items):
                        secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                        secondary_product_moles = secondary_product * self.param["product"]["secondary"]["moles_per_cell"]
                        secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]
                        secondary_product_eq_mat_moles = primary_product * self.param["active_element"]["secondary"][
                            "eq_matrix_moles_per_cell"]

            self.conn.commit()

            # n_matrix_page = (self.axlim ** 2) * self.param["product"]["primary"]["oxidation_number"]
            n_matrix_page = (self.axlim ** 2)
            matrix = np.full(self.axlim, n_matrix_page)

            matrix_moles = matrix * self.param["matrix_elem"]["moles_per_cell"] - outward_eq_mat_moles \
                           - soutward_eq_mat_moles - primary_product_eq_mat_moles - secondary_product_eq_mat_moles \
                           - ternary_product_eq_mat_moles - quaternary_product_eq_mat_moles
            matrix_mass = matrix_moles * self.param["matrix_elem"]["molar_mass"]

            # matrix = (n_matrix_page - outward - soutward -
            #           primary_product - secondary_product - ternary_product - quaternary_product)
            # less_than_zero = np.where(matrix < 0)[0]
            # matrix[less_than_zero] = 0

            # matrix_moles = matrix * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
            # matrix_mass = matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

            x = np.linspace(0, self.param["size"], self.axlim)

            if conc_type.lower() == "atomic":
                whole_moles = matrix_moles + \
                              inward_moles + sinward_moles + \
                              outward_moles + soutward_moles + \
                              primary_product_moles + secondary_product_moles + \
                              ternary_product_moles + quaternary_product_moles

                inward = inward_moles * 100 / whole_moles
                sinward = sinward_moles * 100 / whole_moles
                outward = outward_moles * 100 / whole_moles
                soutward = soutward_moles * 100 / whole_moles

                primary_product = primary_product_moles * 100 / whole_moles
                secondary_product = secondary_product_moles * 100 / whole_moles
                ternary_product = ternary_product_moles * 100 / whole_moles
                quaternary_product = quaternary_product_moles * 100 / whole_moles

            elif conc_type.lower() == "cells":
                n_cells_page = self.axlim ** 2
                inward = inward * 100 / n_cells_page
                sinward = sinward * 100 / n_cells_page
                outward = outward * 100 / n_cells_page
                soutward = soutward * 100 / n_cells_page

                primary_product = primary_product * 100 / n_cells_page
                secondary_product = secondary_product * 100 / n_cells_page
                ternary_product = ternary_product * 100 / n_cells_page
                quaternary_product = quaternary_product * 100 / n_cells_page

            elif conc_type.lower() == "mass":
                whole_mass = matrix_mass + \
                             inward_mass + sinward_mass + \
                             outward_mass + soutward_mass + \
                             secondary_product_mass + primary_product_mass + \
                             ternary_product_mass + quaternary_product_mass

                inward = inward_mass * 100 / whole_mass
                sinward = sinward_mass * 100 / whole_mass
                outward = outward_mass * 100 / whole_mass
                soutward = soutward_mass * 100 / whole_mass

                primary_product = primary_product_mass * 100 / whole_mass
                secondary_product = secondary_product_mass * 100 / whole_mass
                ternary_product = ternary_product_mass * 100 / whole_mass
                quaternary_product = quaternary_product_mass * 100 / whole_mass

            else:
                print("WRONG CONCENTRATION TYPE!")

            ax1.cla()
            ax2.cla()
            ax1.plot(x, inward, color='b')
            ax1.plot(x, sinward, color='deeppink')

            ax2.plot(x, outward, color='g')
            ax2.plot(x, soutward, color='darkorange')

            ax2.plot(x, primary_product, color='r')
            ax2.plot(x, secondary_product, color='cyan')
            ax2.plot(x, ternary_product, color='darkgreen')
            ax2.plot(x, quaternary_product, color='steelblue')

            if analytic_sol:
                y_max = self.param["oxidant"]["primary"]["cells_concentration"] * 100
                # y_max_out = self.param["active_elem_conc"] * 100

                diff_c = self.param["oxidant"]["primary"]["diffusion_coefficient"]

                analytical_concentration_maxy =\
                    y_max * special.erfc(x / (2 * sqrt(diff_c * (iteration + 1) * self.param["sim_time"] /
                                                       self.param["n_iterations"])))
                ax1.plot(x, analytical_concentration_maxy, color='r')

                # analytical_concentration_out = (y_max_out/2) * (1 - special.erf((- x) / (2 * sqrt(
                #     self.param["diff_coeff_out"] * (iteration + 1) * self.param["sim_time"] / self.param["n_iterations"]))))

                # proz = [sqrt((analytic - outw)**2) / analytic for analytic, outw in zip(analytical_concentration_out, outward)]
                # proz_mean = (np.sum(proz[0:10]) / 10) * 100
                # summa = analytical_concentration_out - outward
                # summa = np.sum(summa[0:10])
                # print(f"""{iteration} {proz_mean}""")

                # ax1.set_ylim(0, y_max_out + y_max_out * 0.2)
                # ax1.plot(x, analytical_concentration_out, color='r', linewidth=1.5)
            # if analytic_sol_sand:
            #     self.c.execute("SELECT y_max_sand from description")
            #     y_max_sand = self.c.fetchone()[0] / 2
            #     self.c.execute("SELECT half_thickness from description")
            #     half_thickness = self.c.fetchone()[0]
            #     # left = ((self.n_cells_per_axis / 2) - half_thickness) * self.lamda - self.lamda
            #     # right = ((self.n_cells_per_axis / 2) + half_thickness) * self.lamda + self.lamda
            #
            #     #  for point!
            #     # left = int(self.n_cells_per_axis / 2) * self.lamda
            #     # right = (int(self.n_cells_per_axis / 2) + half_thickness) * self.lamda
            #
            #     left = (int(self.param["n_cells_per_axis"]n_cells_per_axis / 2) - half_thickness) * self.param["l_ambda"]
            #     right = (int(self.param["n_cells_per_axis"]n_cells_per_axis / 2) + half_thickness) * self.param["l_ambda"]
            #     analytical_concentration_sand = \
            #         [y_max_sand *
            #          (special.erf((item - left) / (2 * sqrt(self.param["n_cells_per_axis"]d_coeff_out * (iteration + 1) * self.param["n_cells_per_axis"]time_total /
            #                                                 self.param["n_cells_per_axis"]number_of_iterations))) -
            #           special.erf((item - right) / (2 * sqrt(self.param["n_cells_per_axis"]d_coeff_out * (iteration + 1) * self.param["n_cells_per_axis"]time_total /
            #                                                  self.param["n_cells_per_axis"]number_of_iterations))))
            #          for item in x]
            #     ax1.set_ylim(0, y_max_sand * 2 + y_max_sand * 0.2)
            #     ax1.plot(x, analytical_concentration_sand, color='k')

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        animation = FuncAnimation(fig, animate)
        plt.show()
        self.conn.commit()

    def plot_concentration(self, plot_separate=True, iteration=None, conc_type="atomic", analytic_sol=False):
        if iteration is None:
            iteration = self.last_i

        inward = np.zeros(self.axlim, dtype=int)
        inward_moles = np.zeros(self.axlim, dtype=int)
        inward_mass = np.zeros(self.axlim, dtype=int)

        sinward = np.zeros(self.axlim, dtype=int)
        sinward_moles = np.zeros(self.axlim, dtype=int)
        sinward_mass = np.zeros(self.axlim, dtype=int)

        outward = np.zeros(self.axlim, dtype=int)
        outward_moles = np.zeros(self.axlim, dtype=int)
        outward_mass = np.zeros(self.axlim, dtype=int)
        outward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        soutward = np.zeros(self.axlim, dtype=int)
        soutward_moles = np.zeros(self.axlim, dtype=int)
        soutward_mass = np.zeros(self.axlim, dtype=int)
        soutward_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        primary_product = np.zeros(self.axlim, dtype=int)
        primary_product_moles = np.zeros(self.axlim, dtype=int)
        primary_product_mass = np.zeros(self.axlim, dtype=int)
        primary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        secondary_product = np.zeros(self.axlim, dtype=int)
        secondary_product_moles = np.zeros(self.axlim, dtype=int)
        secondary_product_mass = np.zeros(self.axlim, dtype=int)
        secondary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        ternary_product = np.zeros(self.axlim, dtype=int)
        ternary_product_moles = np.zeros(self.axlim, dtype=int)
        ternary_product_mass = np.zeros(self.axlim, dtype=int)
        ternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        quaternary_product = np.zeros(self.axlim, dtype=int)
        quaternary_product_moles = np.zeros(self.axlim, dtype=int)
        quaternary_product_mass = np.zeros(self.axlim, dtype=int)
        quaternary_product_eq_mat_moles = np.zeros(self.axlim, dtype=int)

        if self.param["inward_diffusion"]:
            self.c.execute("SELECT * from primary_oxidant_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            inward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            inward_moles = inward * self.param["oxidant"]["primary"]["moles_per_cell"]
            inward_mass = inward * self.param["oxidant"]["primary"]["mass_per_cell"]

            if self.param["secondary_oxidant_exists"]:
                self.c.execute("SELECT * from secondary_oxidant_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                sinward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                sinward_moles = sinward * self.param["oxidant"]["secondary"]["moles_per_cell"]
                sinward_mass = sinward * self.param["oxidant"]["secondary"]["mass_per_cell"]

        if self.param["outward_diffusion"]:
            self.c.execute("SELECT * from primary_active_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            outward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            outward_moles = outward * self.param["active_element"]["primary"]["moles_per_cell"]
            outward_mass = outward * self.param["active_element"]["primary"]["mass_per_cell"]
            outward_eq_mat_moles = outward * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
            if self.param["secondary_active_element_exists"]:
                self.c.execute("SELECT * from secondary_active_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                soutward = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                soutward_moles = soutward * self.param["active_element"]["secondary"]["moles_per_cell"]
                soutward_mass = soutward * self.param["active_element"]["secondary"]["mass_per_cell"]
                soutward_eq_mat_moles = soutward * self.param["active_element"]["secondary"]["eq_matrix_moles_per_cell"]

        if self.param["compute_precipitations"]:
            self.c.execute("SELECT * from primary_product_iter_{}".format(iteration))
            items = np.array(self.c.fetchall())
            primary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
            primary_product_moles = primary_product * self.param["product"]["primary"]["moles_per_cell"]
            primary_product_mass = primary_product * self.param["product"]["primary"]["mass_per_cell"]
            primary_product_eq_mat_moles = primary_product * self.param["active_element"]["primary"][
                "eq_matrix_moles_per_cell"]

            if self.param["secondary_active_element_exists"] and self.param["secondary_oxidant_exists"]:
                self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                secondary_product_moles = secondary_product * self.param["product"]["secondary"]["moles_per_cell"]
                secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]
                secondary_product_eq_mat_moles = secondary_product * self.param["active_element"]["secondary"][
                    "eq_matrix_moles_per_cell"]

                self.c.execute("SELECT * from ternary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                ternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                ternary_product_moles = ternary_product * self.param["product"]["ternary"]["moles_per_cell"]
                ternary_product_mass = ternary_product * self.param["product"]["ternary"]["mass_per_cell"]
                ternary_product_eq_mat_moles = ternary_product * self.param["active_element"]["primary"][
                    "eq_matrix_moles_per_cell"]

                self.c.execute("SELECT * from quaternary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                quaternary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                quaternary_product_moles = quaternary_product * self.param["product"]["quaternary"]["moles_per_cell"]
                quaternary_product_mass = quaternary_product * self.param["product"]["quaternary"]["mass_per_cell"]
                quaternary_product_eq_mat_moles = quaternary_product * self.param["active_element"]["secondary"][
                    "eq_matrix_moles_per_cell"]

            elif self.param["secondary_active_element_exists"] and not self.param["secondary_oxidant_exists"]:
                self.c.execute("SELECT * from secondary_product_iter_{}".format(iteration))
                items = np.array(self.c.fetchall())
                if np.any(items):
                    secondary_product = np.array([len(np.where(items[:, 2] == i)[0]) for i in range(self.axlim)])
                    secondary_product_moles = secondary_product * self.param["product"]["secondary"]["moles_per_cell"]
                    secondary_product_mass = secondary_product * self.param["product"]["secondary"]["mass_per_cell"]
                    secondary_product_eq_mat_moles = primary_product * self.param["active_element"]["secondary"][
                        "eq_matrix_moles_per_cell"]

        self.conn.commit()

        # n_matrix_page = (self.axlim ** 2) * self.param["product"]["primary"]["oxidation_number"]
        n_matrix_page = (self.axlim ** 2)
        matrix = np.full(self.axlim, n_matrix_page)

        matrix_moles = matrix * self.param["matrix_elem"]["moles_per_cell"] - outward_eq_mat_moles\
                       - soutward_eq_mat_moles - primary_product_eq_mat_moles - secondary_product_eq_mat_moles\
                       - ternary_product_eq_mat_moles - quaternary_product_eq_mat_moles
        matrix_mass = matrix_moles * self.param["matrix_elem"]["molar_mass"]

        # matrix = (n_matrix_page - outward - soutward -
        #           primary_product - secondary_product - ternary_product - quaternary_product)
        # less_than_zero = np.where(matrix < 0)[0]
        # matrix[less_than_zero] = 0

        # matrix_moles = matrix * self.param["active_element"]["primary"]["eq_matrix_moles_per_cell"]
        # matrix_mass = matrix * self.param["active_element"]["primary"]["eq_matrix_mass_per_cell"]

        x = np.linspace(0, self.param["size"], self.axlim)

        if conc_type.lower() == "atomic":
            whole_moles = matrix_moles +\
                          inward_moles + sinward_moles +\
                          outward_moles + soutward_moles +\
                          primary_product_moles + secondary_product_moles +\
                          ternary_product_moles + quaternary_product_moles

            inward = inward_moles * 100 / whole_moles
            sinward = sinward_moles * 100 / whole_moles
            outward = outward_moles * 100 / whole_moles
            soutward = soutward_moles * 100 / whole_moles

            primary_product = primary_product_moles * 100 / whole_moles
            secondary_product = secondary_product_moles * 100 / whole_moles
            ternary_product = ternary_product_moles * 100 / whole_moles
            quaternary_product = quaternary_product_moles * 100 / whole_moles

        elif conc_type.lower() == "cells":
            n_cells_page = self.axlim ** 2
            inward = inward * 100 / n_cells_page
            sinward = sinward * 100 / n_cells_page
            outward = outward * 100 / n_cells_page
            soutward = soutward * 100 / n_cells_page

            primary_product = primary_product * 100 / n_cells_page
            secondary_product = secondary_product * 100 / n_cells_page
            ternary_product = ternary_product * 100 / n_cells_page
            quaternary_product = quaternary_product * 100 / n_cells_page

        elif conc_type.lower() == "mass":
            whole_mass = matrix_mass +\
                         inward_mass + sinward_mass +\
                         outward_mass + soutward_mass +\
                         secondary_product_mass + primary_product_mass +\
                         ternary_product_mass + quaternary_product_mass

            inward = inward_mass * 100 / whole_mass
            sinward = sinward_mass * 100 / whole_mass
            outward = outward_mass * 100 / whole_mass
            soutward = soutward_mass * 100 / whole_mass

            primary_product = primary_product_mass * 100 / whole_mass
            secondary_product = secondary_product_mass * 100 / whole_mass
            ternary_product = ternary_product_mass * 100 / whole_mass
            quaternary_product = quaternary_product_mass * 100 / whole_mass

        else:
            print("WRONG CONCENTRATION TYPE!")

        fig = plt.figure()
        if plot_separate:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(x, inward, color='b')
            ax1.plot(x, sinward, color='deeppink')

            ax2.plot(x, outward, color='g')
            ax2.plot(x, soutward, color='darkorange')

            ax2.plot(x, primary_product, color='r')
            ax2.plot(x, secondary_product, color='cyan')
            ax2.plot(x, ternary_product, color='darkgreen')
            ax2.plot(x, quaternary_product, color='steelblue')

            if analytic_sol:
                if conc_type == "atomic":
                    y_max = max(inward)
                    y_max_out = self.param["active_element"]["primary"]["atomic_concentration"] * 100

                elif conc_type == "cells":
                    y_max = self.param["oxidant"]["primary"]["cells_concentration"] * 100
                    y_max_out = self.param["active_element"]["primary"]["cells_concentration"] * 100

                elif conc_type == "mass":
                    y_max = max(inward)
                    y_max_out = self.param["active_element"]["primary"]["mass_concentration"] * 100

                diff_in = self.param["oxidant"]["primary"]["diffusion_coefficient"]
                diff_out = self.param["active_element"]["primary"]["diffusion_coefficient"]

                analytical_concentration = y_max * special.erfc(x / (2 * sqrt(diff_in * self.param["sim_time"])))
                analytical_concentration_out = (y_max_out / 2) * (1 - special.erf((- x + 0.0005) / (2 * sqrt(
                    diff_out * (iteration + 1) * self.param["sim_time"] / self.param["n_iterations"]))))

                # ax1.set_ylim(0, y_max + y_max * 0.2)
                # ax2.set_ylim(0, y_max_out + y_max_out * 0.2)
                ax2.plot(x, analytical_concentration_out, color='r', linewidth=1.5)
                ax1.plot(x, analytical_concentration, color='r', linewidth=1.5)
        else:
            ax = fig.add_subplot(111)

            ax.plot(x, inward, color='b')
            ax.plot(x, sinward, color='deeppink')
            ax.plot(x, outward, color='g')
            ax.plot(x, soutward, color='darkorange')

            ax.plot(x, primary_product, color='r')
            ax.plot(x, secondary_product, color='cyan')
            ax.plot(x, ternary_product, color='darkgreen')
            ax.plot(x, quaternary_product, color='steelblue')

            ax.set_xlabel("Depth [m]")
            ax.set_ylabel("Concentration")

            # ax.plot(x, outward,  color='g')
            # ax.plot(x, precipitations,  color='r')

            if analytic_sol:
                y_max = self.param["diff_elem_conc"] * 100
                analytical_concentration = y_max * special.erfc(x / (2 * sqrt(self.param["diff_coeff_in"] *
                                                                              self.param["sim_time"])))
                ax.set_ylim(0, y_max + y_max * 0.2)
                ax.plot(x, analytical_concentration, color='r', linewidth=1.5)

            # if analytic_sol_sand:
            #     self.c.execute("SELECT y_max_sand from description")
            #     y_max_sand = self.c.fetchone()[0] / 2
            #     self.c.execute("SELECT half_thickness from description")
            #     half_thickness = self.c.fetchone()[0]
            #     left = (int(self.param["inward_diffusion"]n_cells_per_axis / 2) - half_thickness) * self.param["inward_diffusion"]lamda
            #     right = (int(self.param["inward_diffusion"]n_cells_per_axis / 2) + half_thickness) * self.param["inward_diffusion"]lamda
            #     analytical_concentration_sand = \
            #         [y_max_sand *
            #          (special.erf(
            #              (item - left) / (2 * sqrt(self.param["inward_diffusion"]d_coeff_out * (iteration + 1) * self.param["inward_diffusion"]time_total /
            #                                        self.param["inward_diffusion"]number_of_iterations))) -
            #           special.erf(
            #               (item - right) / (2 * sqrt(self.param["inward_diffusion"]d_coeff_out * (iteration + 1) * self.param["inward_diffusion"]time_total /
            #                                          self.param["inward_diffusion"]number_of_iterations))))
            #          for item in x]
            #     ax.set_ylim(0, y_max_sand * 2 + y_max_sand * 0.2)
            #     ax.plot(x, analytical_concentration_sand, color='k')
        # plt.savefig(f'{self.db_name}_{iteration}.jpeg')
        plt.show()

    def plot_h(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        self.conn = sql.connect(self.db_name)
        self.c = self.conn.cursor()

        self.c.execute("SELECT * from precip_front_p")
        items = np.array(self.c.fetchall())
        if np.any(items):
            sqr_time = items[:, 0]
            position = items[:, 1]
            ax1.scatter(sqr_time, position, s=10, color='r')
        else:
            return print("No Data to plot primary precipitation front!")

        if self.param["secondary_active_element_exists"]:
            self.c.execute("SELECT * from precip_front_s")
            items = np.array(self.c.fetchall())
            if np.any(items):
                sqr_time_s = items[:, 0]
                position_s = items[:, 1]
                ax1.scatter(sqr_time_s, position_s, s=10, color='cyan')
            else:
                return print("No Data to plot secondary precipitation front!")
        plt.show()

