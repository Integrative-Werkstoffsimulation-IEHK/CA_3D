from configuration import Config
from utils import new_utils
import sqlite3
import pickle
import tkinter as tk
from tkinter import filedialog

#
# new_util = new_utils.NUtils()
# new_util.generate_param()
# new_util.create_database()
# new_util.db.conn.commit()
# new_util.db.conn.close()

root = tk.Tk()
root.withdraw()
database_name = filedialog.askopenfilename()

# Connect to SQLite database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Query the pickled data from the database
cursor.execute("SELECT pickled_data FROM PickledConfig")
result = cursor.fetchone()

if result:
    # Unpickle the data
    pickled_instance = result[0]
    config_instance = pickle.loads(pickled_instance)

    # Use the instantiated class
else:
    print("No pickled data found in the database.")

# Close connection
conn.close()


root = tk.Tk()
root.withdraw()
database_name = filedialog.askopenfilename()

# Connect to the SQLite database
conn = sqlite3.connect(database_name)
cursor = conn.cursor()

# Define the table name you want to check
table_name = 'PickledConfig'

# Check if the table exists
cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
result = cursor.fetchone()

# If the table doesn't exist, perform some other actions
if result is None:
    print("Table does not exist, performing other actions...")
    # Perform other actions here, like creating the table or any other operation
else:
    print("Table exists.")

# Close the connection
conn.close()
