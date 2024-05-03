import sqlite3
import pickle
import tkinter as tk
from tkinter import filedialog
import json
from configuration import Config

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


def get_static_vars_dict(cls):
    # Initialize an empty dictionary
    static_vars_dict = {}
    # Iterate over class attributes
    for attr_name, attr_value in cls.__dict__.items():
        # Exclude special methods and variables starting with '__'
        if not attr_name.startswith('__') and not isinstance(attr_value, classmethod):
            # If the attribute is an instance of another class, recursively convert it to a dictionary
            if isinstance(attr_value, type):
                static_vars_dict[attr_name] = get_static_vars_dict(attr_value)
            else:
                static_vars_dict[attr_name] = attr_value
    return static_vars_dict


def update_class_from_dict(cls, data):
    for key, value in data.items():
        if isinstance(value, dict):
            setattr(cls, key, type(key, (), value))
        else:
            setattr(cls, key, value)


Config.COMMENT = """sgsfg
sfgsdfg
fg"""
Config.GENERATED_VALUES.new_val = 1234
Config.PRODUCTS.PRIMARY.OXID_NUMB = 23

dict_to_pickle = get_static_vars_dict(Config)
pickled_instance = pickle.dumps(dict_to_pickle)

# Connect to SQLite database
conn = sqlite3.connect("test_db")
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS PickledConfig (pickled_data BLOB)''')
cursor.execute("INSERT INTO PickledConfig (pickled_data) VALUES (?)", (pickled_instance,))
conn.commit()
conn.close()


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
