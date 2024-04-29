from config import Config
import sqlite3
import pickle

conf_instance = Config()

# Pickle the instance
pickled_instance = pickle.dumps(conf_instance)

# Connect to SQLite database
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Create a table to store pickled data

cursor.execute('''CREATE TABLE IF NOT EXISTS PickledData
                (pickled_data BLOB)''')

# Insert pickled data into the table
cursor.execute("INSERT INTO PickledData (pickled_data) VALUES (?)", (pickled_instance,))

# Commit changes and close connection
conn.commit()
conn.close()

# Connect to SQLite database
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Query the pickled data from the database
cursor.execute("SELECT pickled_data FROM PickledData")
result = cursor.fetchone()

if result:
    # Unpickle the data
    pickled_instance = result[0]
    config_instance = pickle.loads(pickled_instance)

    # Use the instantiated class
    print()
else:
    print("No pickled data found in the database.")

# Close connection
conn.close()

