import numpy as np


class TDATA:
    def __init__(self):
        self.lookup_table = {}
        # self.gen_table()

    def gen_table_nesed_dict(self):
        # Define the composition ranges for Cr, Al, and O
        cr_range = np.around(np.linspace(0, 25, 101), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 2.5, 101), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 0.001, 101), decimals=8)  # Example O composition values

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
        cr_range = np.around(np.linspace(0, 26.585, 11), decimals=4)  # Example Cr composition values
        al_range = np.around(np.linspace(0, 6, 11), decimals=4)  # Example Al composition values
        o_range = np.around(np.linspace(0, 0.001, 101), decimals=8)  # Example O composition values

        # Populate the lookup table with probabilities
        for cr in cr_range:
            for al in al_range:
                for o in o_range:
                    if cr == 0 and al == 0:
                        continue
                    elif o == 0:
                        continue
                    else:
                        # Assign the corresponding probability for the composition values
                        # self.lookup_table[cr, al, o] = 1.0
                        print(f"{cr} {al} {o:.5f}")


if __name__ == '__main__':
    test_data = TDATA()
    test_data.gen_table_dict()
    print()
