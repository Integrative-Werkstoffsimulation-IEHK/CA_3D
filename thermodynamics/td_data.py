import numpy as np


class Component:
    def __init__(self, constitution, amount):
        self.constitution = constitution
        self.amount = amount


class CompPool:
    def __init__(self):
        self.primary = None
        self.secondary = None
        self.ternary = None
        self.quaternary = None


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
                    if cr == 0 and al == 0 and o == 0:
                        continue
                    elif sum_conc > 100:
                        continue
                    elif al >= cr and o != 0:
                        # Assign the corresponding probability for the composition values
                        new_comp_pool = CompPool()
                        new_comp_pool.primary = Component("Al2O3", 1)
                        new_comp_pool.secondary = Component("Cr2O3", 0)
                        self.lookup_table[cr, al, o, sum_conc] = new_comp_pool
                        # print(f"{cr} {al} {o:.5f}")
                    elif cr > al and o != 0:
                        new_comp_pool = CompPool()
                        new_comp_pool.primary = Component("Al2O3", 0)
                        new_comp_pool.secondary = Component("Cr2O3", 1)
                        self.lookup_table[cr, al, o, sum_conc] = new_comp_pool


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
            p_val = value.primary.constitution
            p_val_a = value.primary.amount

            s_val = value.secondary.constitution
            s_val_a = value.secondary.amount

            file.write(f"{column1} {column2} {column3} {sum_conc} {p_val}:{p_val_a} {s_val}:{s_val_a}\n")
