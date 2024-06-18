import sqlite3
import pickle
import tkinter as tk
from tkinter import filedialog
import json
from configuration import Config
import multiprocessing
from engine import *
import traceback


# #
# # new_util = new_utils.NUtils()
# # new_util.generate_param()
# # new_util.create_database()
# # new_util.db.conn.commit()
# # new_util.db.conn.close()
#

# root = tk.Tk()
# root.withdraw()
# database_name = filedialog.askopenfilename()
#
# # Connect to SQLite database
# conn = sqlite3.connect(database_name)
# cursor = conn.cursor()
#
# # Query the pickled data from the database
# cursor.execute("SELECT pickled_data FROM PickledConfig")
# result = cursor.fetchone()
#
# if result:
#     # Unpickle the data
#     pickled_instance = result[0]
#     config_instance = pickle.loads(pickled_instance)
#
#     # Use the instantiated class
# else:
#     print("No pickled data found in the database.")
#
# # Close connection
# conn.close()
#
#
# root = tk.Tk()
# root.withdraw()
# database_name = filedialog.askopenfilename()
#
# # Connect to the SQLite database
# conn = sqlite3.connect(database_name)
# cursor = conn.cursor()
#
# # Define the table name you want to check
# table_name = 'PickledConfig'
#
# # Check if the table exists
# cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
# result = cursor.fetchone()
#
# # If the table doesn't exist, perform some other actions
# if result is None:
#     print("Table does not exist, performing other actions...")
#     # Perform other actions here, like creating the table or any other operation
# else:
#     print("Table exists.")
#
# # Close the connection
# conn.close()


# def get_static_vars_dict(cls):
#     # Initialize an empty dictionary
#     static_vars_dict = {}
#     # Iterate over class attributes
#     for attr_name, attr_value in cls.__dict__.items():
#         # Exclude special methods and variables starting with '__'
#         if not attr_name.startswith('__') and not isinstance(attr_value, classmethod):
#             # If the attribute is an instance of another class, recursively convert it to a dictionary
#             if isinstance(attr_value, type):
#                 static_vars_dict[attr_name] = get_static_vars_dict(attr_value)
#             else:
#                 static_vars_dict[attr_name] = attr_value
#     return static_vars_dict
#
#
# def update_class_from_dict(cls, data):
#     for key, value in data.items():
#         if isinstance(value, dict):
#             setattr(cls, key, type(key, (), value))
#         else:
#             setattr(cls, key, value)
#
#
# Config.COMMENT = """sgsfg
# sfgsdfg
# fg"""
# Config.GENERATED_VALUES.new_val = 1234
# Config.PRODUCTS.PRIMARY.OXID_NUMB = 23
#
# dict_to_pickle = get_static_vars_dict(Config)
# pickled_instance = pickle.dumps(dict_to_pickle)
#
# # Connect to SQLite database
# conn = sqlite3.connect("test_db")
# cursor = conn.cursor()
#
# cursor.execute('''CREATE TABLE IF NOT EXISTS PickledConfig (pickled_data BLOB)''')
# cursor.execute("INSERT INTO PickledConfig (pickled_data) VALUES (?)", (pickled_instance,))
# conn.commit()
# conn.close()


# # Connect to SQLite database
# conn = sqlite3.connect("test_db")
# cursor = conn.cursor()
#
# table_name = 'PickledConfig'
# cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
# result = cursor.fetchone()
#
# if result is None:
#     print()
# else:
#     cursor.execute("SELECT pickled_data FROM PickledConfig")
#     result = cursor.fetchone()
#     # Unpickle the data
#     unpickled_instance = result[0]
#     n_dict = pickle.loads(unpickled_instance)
#
#     update_class_from_dict(Config, n_dict)
#     print()


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # Create a figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Define the grid for the plane
# x = np.linspace(-5, 5, 10)
# y = np.linspace(-5, 5, 10)
# X, Y = np.meshgrid(x, y)
#
# # Define the Z coordinates for the plane
# Z = np.full(X.shape, 2)  # This sets Z = 2 for the entire plane, making it parallel to the YX axis
#
# # Plot the plane
# ax.plot_surface(X, Y, Z, color='cyan', alpha=0.5)  # Set alpha to a value between 0 and 1 for transparency
#
# # Optionally, add more elements to the plot for context
# ax.scatter([0], [0], [2], color='red', s=100)  # Example point to show position relative to the plane
#
#
# # Show the plot
# plt.show()

# import numpy as np
# import timeit
#
# # Creating a large sample array for performance testing
# array = np.random.rand(100000, 10000)
#
# def method_1():
#     temp = array[:, 0].copy()
#     array[:, 0] = array[:, 2]
#     array[:, 2] = temp
#
# def method_2():
#     array[:, [0, 2]] = array[:, [2, 0]].copy()
#
# def method_3():
#     array[:, [0, 2]] = array[:, [2, 0]]
#
# # Measuring the execution time of each method
# time = 0
# for _ in range(10):
#     time += timeit.timeit(method_1, number=100)
# print(f"Method 1 time: {time/10}")
#
# time = 0
# for _ in range(10):
#     time += timeit.timeit(method_2, number=100)
# print(f"Method 2 time: {time/10}")
#
# time = 0
# for _ in range(10):
#     time += timeit.timeit(method_3, number=100)
# print(f"Method 3 time: {time/10}")


from engine import *
import traceback
from configuration import Config


def single_proc():
        print("Started")
        Config.COMMENT = """
    
        eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
        eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale
    
        eng.precip_func = eng.precipitation_first_case
        eng.get_combi_ind = eng.get_combi_ind_atomic_with_kinetic
        eng.precip_step = eng.precip_step_standard
        eng.check_intersection = eng.ci_single
    
        eng.decomposition = eng.dissolution_atomic_with_kinetic
        eng.decomposition_intrinsic = eng.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL
    
        eng.cur_case = eng.cases.first
        eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip
    
        eng.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                              Config.PRODUCTS.PRIMARY)
        eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.PRIMARY)
    
        Script name: main.py
        Nucleation and dissolution throughout the whole simulation (both schemes applied). Also with kinetic coeeficient!!!
        Go along the kinetic growth line! Check the kinetic file as well!!!
    
        CHANGED THE SCHEMES OF NUCLEATION AND DISSOLUTION ->>> NOW ALSO THE PARTIAL NEIGHBOURS ARE CONSIDERED!!!!
    
    """

        eng = CellularAutomata()

        eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
        eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale

        eng.precip_func = eng.precipitation_first_case
        eng.get_combi_ind = eng.get_combi_ind_atomic_with_kinetic
        eng.precip_step = eng.precip_step_standard
        eng.check_intersection = eng.ci_single

        eng.decomposition = eng.dissolution_atomic_with_kinetic
        eng.decomposition_intrinsic = eng.dissolution_zhou_wei_with_bsf_aip_UPGRADE_BOOL

        eng.cur_case = eng.cases.first
        eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip

        eng.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                              Config.PRODUCTS.PRIMARY)
        eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                                Config.PRODUCTS.PRIMARY)

        try:
            eng.simulation()
        finally:
            try:
                if not Config.SAVE_WHOLE:
                    eng.save_results()

            except (Exception,):
                eng.save_results()
                print("Not SAVED!")
            #     backup_user_input["save_path"] = "C:/test_runs_data/"
            #     eng.utils = Utils(backup_user_input)
            #     eng.utils.create_database()
            #     eng.utils.generate_param()
            #     eng.save_results()
            #     print()
            #     print("____________________________________________________________")
            #     print("Saving To Standard Folder Crashed!!!")
            #     print("Saved To ->> C:/test_runs_data/!!!")
            #     print("____________________________________________________________")
            #     print()
            #
            iterations = np.arange(eng.cumul_prod.last_in_buffer) * eng.precipitation_stride

            data = np.column_stack(
                (iterations, eng.cumul_prod.get_buffer(), eng.growth_rate.get_buffer()))
            output_file_path = "C:/test_runs_data/" + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
            with open(output_file_path, "w") as f:
                for row in data:
                    f.write(" ".join(map(str, row)) + "\n")

            eng.insert_last_it()
            eng.utils.db.conn.commit()
            print()
            print("____________________________________________________________")
            print("Simulation was closed at Iteration: ", eng.iteration)
            print("____________________________________________________________")
            print()
            traceback.print_exc()

            print("Finished")


if __name__ == '__main__':
    # Create process objects
    process1 = multiprocessing.Process(target=single_proc)
    process2 = multiprocessing.Process(target=single_proc)

    # Start the processes
    process1.start()
    process2.start()

    # Wait for the processes to complete
    process1.join()
    process2.join()

