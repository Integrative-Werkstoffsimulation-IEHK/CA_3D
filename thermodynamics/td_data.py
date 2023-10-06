import numpy as np


class TDATA:
    def __init__(self):
        self.lookup_table = {}
        # self.gen_table()

    def gen_table_nesed_dict(self):
        # Define the composition ranges for Cr, Al, and O
        cr_range = np.around(np.linspace(0, 25, 10), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 2.5, 10), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 60, 10), decimals=8)  # Example O composition values

        # Populate the lookup table with probabilities
        for cr in cr_range:
            self.lookup_table[cr] = {}  # Create nested dictionary for Cr
            for al in al_range:
                self.lookup_table[cr][al] = {}  # Create nested dictionary for Al
                for o in o_range:
                    # Assign the corresponding probability for the composition values
                    self.lookup_table[cr][al][o] = 5.1

    def gen_table_dict(self):
        # Define the composition ranges for Cr, Al, and O
        cr_range = np.around(np.linspace(0, 40, 101), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 40, 101), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 60, 101), decimals=8)  # Example O composition values

        # Populate the lookup table with probabilities
        for cr in cr_range:
            for al in al_range:
                for o in o_range:
                    sum_conc = cr + al + o
                    if cr == 0 or al == 0 or o == 0:
                        continue
                    elif sum_conc > 100:
                        continue
                    else:
                        # Assign the corresponding probability for the composition values
                        self.lookup_table[cr, al, o, sum_conc] = 1.0
                        # print(f"{cr} {al} {o:.5f}")


if __name__ == '__main__':
    test_data = TDATA()
    test_data.gen_table_dict()

    # Open a text file for writing
    with open("TD_table_output.txt", "w") as file:
        # Write a header line if needed
        file.write("Cr Al O Sum_Conc Value\n")

        # Iterate through the dictionary items and write them to the file
        for key, value in test_data.lookup_table.items():
            column1, column2, column3, sum_conc = key
            file.write(f"{column1} {column2} {column3} {sum_conc} {value}\n")

