# PRIMARY = {
#
#     "p0": 0.1,
#     "p0_f": 1,
#     "p0_A_const": 1,
#     "p0_B_const": 1,
#
#     "p1": 0.5,
#     "p1_f": 1,
#     "p1_A_const": 1,
#     "p1_B_const": 1,
#
#     "global_A": 1,
#     "global_B": 0.07,
#     "global_B_f": -20,
#
#     "max_neigh_numb": 0,
#
#     "p0_d": 0.01,
#     "p0_d_f": 1,
#     "p0_d_A_const": 1,
#     "p0_d_B_const": 1,
#
#     "p1_d": 0.001,
#     "p1_d_f": 1,
#     "p1_d_A_const": 1,
#     "p1_d_B_const": 1,
#
#     "p6_d": 1*10**-10,
#     "p6_d_f": 0.1,
#     "p6_d_A_const": 1,
#     "p6_d_B_const": 1,
#
#
#     "global_d_A": 1,
#     "global_d_B": -2,
#     "global_d_B_f": -0.001,
#
#     "bsf": 1,
#
#     "nucl_adapt_function": 3,
#     "dissol_adapt_function": 3,
#
#
#                   }
#
# SECONDARY = {
#
#     "p0": 0.5,
#     "p0_f": 1,
#     "p0_A_const": 1,
#     "p0_B_const": 1,
#
#     "p1": 0.8,
#     "p1_f": 1,
#     "p1_A_const": 1,
#     "p1_B_const": 1,
#
#     "global_A": 1,
#     "global_B": 0.15,
#     "global_B_f": -20,
#
#     "max_neigh_numb": 0,
#
#     "p0_d": 0.01,
#     "p0_d_f": 1,
#     "p0_d_A_const": 1,
#     "p0_d_B_const": 1,
#
#     "p1_d": 0.001,
#     "p1_d_f": 1,
#     "p1_d_A_const": 1,
#     "p1_d_B_const": 1,
#
#     "p6_d": 1*10**-10,
#     "p6_d_f": 0.1,
#     "p6_d_A_const": 1,
#     "p6_d_B_const": 1,
#
#     "global_d_A": 1,
#     "global_d_B": -4,
#     "global_d_B_f": -0.001,
#
#     "bsf": 1,
#
#     "nucl_adapt_function": 3,
#     "dissol_adapt_function": 3,
#
# }
class ConfigProbabilities:
    def __init__(self):
        # nucleation
        # _________________________
        self.p0 = None
        self.p0_f = None
        self.p0_A_const = None
        self.p0_B_const = None
        self.p1 = None
        self.p1_f = None
        self.p1_A_const = None
        self.p1_B_const = None
        self.global_A = None
        self.global_B = None
        self.global_B_f = None
        self.max_neigh_numb = None
        self.nucl_adapt_function = None
        # _________________________

        # dissolution
        # _________________________
        self.p0_d = None
        self.p0_d_f = None
        self.p0_d_A_const = None
        self.p0_d_B_const = None
        self.p1_d = None
        self.p1_d_f = None
        self.p1_d_A_const = None
        self.p1_d_B_const = None
        self.p6_d = None
        self.p6_d_f = None
        self.p6_d_A_const = None
        self.p6_d_B_const = None
        self.global_d_A = None
        self.global_d_B = None
        self.global_d_B_f = None
        self.bsf = None
        self.dissol_adapt_function = None
        # ________________________


class ElementGroups:
    def __init__(self):
        self.PRIMARY = None
        self.SECONDARY = None


class ElemInput:
    def __init__(self):
        self.ELEMENT = "None"
        self.DIFFUSION_CONDITION = "None"
        self.MASS_CONCENTRATION = "None"
        self.CELLS_CONCENTRATION = "None"

    def __bool__(self):
        return True


class ProdGroups:
    def __init__(self):
        self.PRIMARY = None
        self.SECONDARY = None
        self.TERNARY = None
        self.QUATERNARY = None

    def __bool__(self):
        return True


class ProdInput:
    def __init__(self):
        pass


class GeneratedValues:
    pass


class Config:
    OXIDANTS = ElementGroups()
    ACTIVES = ElementGroups()
    OXIDANTS.PRIMARY = ElemInput()
    ACTIVES.PRIMARY = ElemInput()
    # OXIDANTS.SECONDARY = ElemInput()
    # ACTIVES.SECONDARY = ElemInput()
    MATRIX = ElemInput()

    # oxidants
    # _______________________________________________________________________________
    # primary
    OXIDANTS.PRIMARY.ELEMENT = "O"
    OXIDANTS.PRIMARY.DIFFUSION_CONDITION = "O in Ni Krupp"
    OXIDANTS.PRIMARY.CELLS_CONCENTRATION = 0.01
    # secondary
    # OXIDANTS.SECONDARY.ELEMENT = "None"
    # OXIDANTS.SECONDARY.DIFFUSION_CONDITION = "N in Ni Krupp"
    # OXIDANTS.SECONDARY.CELLS_CONCENTRATION = 0.1
    # _______________________________________________________________________________

    # actives
    # _______________________________________________________________________________
    # primary
    ACTIVES.PRIMARY.ELEMENT = "Cr"
    ACTIVES.PRIMARY.DIFFUSION_CONDITION = "Cr in Ni Krupp"
    ACTIVES.PRIMARY.MASS_CONCENTRATION = 0.11
    ACTIVES.PRIMARY.CELLS_CONCENTRATION = 0.2
    # secondary
    # ACTIVES.SECONDARY.ELEMENT = "None"
    # ACTIVES.SECONDARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    # ACTIVES.SECONDARY.MASS_CONCENTRATION = 0.025
    # ACTIVES.SECONDARY.CELLS_CONCENTRATION = 0.08
    # _______________________________________________________________________________

    # matrix
    # _______________________________________________________________________________
    MATRIX.ELEMENT = "Ni"
    # _______________________________________________________________________________

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 102  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 300000  # must be >= n_cells_per_axis
    STRIDE = 40  # n_iterations / stride = n_iterations for outward diffusion
    SIM_TIME = 72000  # [sek]
    SIZE = 500 * (10 ** -6)  # [m]
    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 0.045

    THRESHOLD_INWARD = 1
    THRESHOL_OUTWARD = 1
    NEIGH_RANGE = 1  # neighbouring ranges    1, 2, 3, 4, 5,  6,  7,  8,  9,  10
                      #          and           |  |  |  |  |   |   |   |   |   |
                      # corresponding divisors 3, 5, 7, 9, 11, 13, 15, 17, 19, 21

    INWARD_DIFFUSION = True
    OUTWARD_DIFFUSION = True
    COMPUTE_PRECIPITATION = True
    SAVE_WHOLE = False
    DECOMPOSE_PRECIPITATIONS = False
    FULL_CELLS = False
    SAVE_PATH = 'C:/test_runs_data/'
    GENERATE_POST_PROCESSED_INPUT = True

    # PROBABILITIES
    # _______________________________________________________________________________
    PROBABILITIES = ElementGroups()
    PROBABILITIES.PRIMARY = ConfigProbabilities()
    # PROBABILITIES.SECONDARY = ConfigProbabilities()
    # PROBABILITIES.TERNARY = ConfigProbabilities()
    # PROBABILITIES.QUATERNARY = ConfigProbabilities()

    # primary
    # nucleation
    # _________________________
    PROBABILITIES.PRIMARY.p0 = 0.1
    PROBABILITIES.PRIMARY.p0_f = 1
    PROBABILITIES.PRIMARY.p0_A_const = 1
    PROBABILITIES.PRIMARY.p0_B_const = 1

    PROBABILITIES.PRIMARY.p1 = 0.5
    PROBABILITIES.PRIMARY.p1_f = 1
    PROBABILITIES.PRIMARY.p1_A_const = 1
    PROBABILITIES.PRIMARY.p1_B_const = 1

    PROBABILITIES.PRIMARY.global_A = 1
    PROBABILITIES.PRIMARY.global_B = 0.07
    PROBABILITIES.PRIMARY.global_B_f = -20

    PROBABILITIES.PRIMARY.max_neigh_numb = None
    PROBABILITIES.PRIMARY.nucl_adapt_function = 3
    # _________________________
    # dissolution
    # _________________________
    PROBABILITIES.PRIMARY.p0_d = 0.01
    PROBABILITIES.PRIMARY.p0_d_f = 1
    PROBABILITIES.PRIMARY.p0_d_A_const = 1
    PROBABILITIES.PRIMARY.p0_d_B_const = 1

    PROBABILITIES.PRIMARY.p1_d = 0.001
    PROBABILITIES.PRIMARY.p1_d_f = 1
    PROBABILITIES.PRIMARY.p1_d_A_const = 1
    PROBABILITIES.PRIMARY.p1_d_B_const = 1

    PROBABILITIES.PRIMARY.p6_d = 1*10**-10
    PROBABILITIES.PRIMARY.p6_d_f = 0.1
    PROBABILITIES.PRIMARY.p6_d_A_const = 1
    PROBABILITIES.PRIMARY.p6_d_B_const = 1

    PROBABILITIES.PRIMARY.global_d_A = 1
    PROBABILITIES.PRIMARY.global_d_B = -2
    PROBABILITIES.PRIMARY.global_d_B_f = -0.001

    PROBABILITIES.PRIMARY.bsf = 1
    PROBABILITIES.PRIMARY.dissol_adapt_function = 3
    # ________________________

    GENERATED_VALUES = GeneratedValues()


def print_static_params(cls, indent=0):
    for attr_name, attr_value in cls.__dict__.items():
        if not callable(attr_value) and not attr_name.startswith('__'):
            if hasattr(attr_value, '__dict__'):
                print(f"{' ' * indent}{attr_name}:")
                print_static_params(attr_value, indent + 4)
            else:
                print(f"{' ' * indent}{attr_name}: {attr_value}")


def print_static_params_to_file(cls, file_obj, indent=0):
    for attr_name, attr_value in cls.__dict__.items():
        if not callable(attr_value) and not attr_name.startswith('__'):
            if hasattr(attr_value, '__dict__'):
                file_obj.write(f"{' ' * indent}{attr_name}:\n")
                print_static_params_to_file(attr_value, file_obj, indent + 4)
            else:
                file_obj.write(f"{' ' * indent}{attr_name}: {attr_value}\n")

with open('output.txt', 'w') as file:
    print_static_params_to_file(Config, file)

# new_conf = Config
# print_static_params(new_conf)
# Config.GENERATED_VALUES.some_new_val = 12
# print_static_params(new_conf)