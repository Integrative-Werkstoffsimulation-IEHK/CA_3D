from .config_utils_classes import *


class Config:
    OXIDANTS = ElementGroups()
    OXIDANTS.PRIMARY = ElemInput()
    OXIDANTS.SECONDARY = ElemInput()
    ACTIVES = ElementGroups()
    ACTIVES.PRIMARY = ElemInput()
    ACTIVES.SECONDARY = ElemInput()
    PRODUCTS = ProdGroups()
    PRODUCTS.PRIMARY = ProdInput()
    PRODUCTS.SECONDARY = ProdInput()
    PRODUCTS.TERNARY = ProdInput()
    PRODUCTS.QUATERNARY = ProdInput()
    MATRIX = ElemInput()

    # USER INPUT FROM HERE______________________________________________________________________________________________
    # primary oxidants
    OXIDANTS.PRIMARY.ELEMENT = "O"
    OXIDANTS.PRIMARY.DIFFUSION_CONDITION = "O in Ni Krupp"
    OXIDANTS.PRIMARY.CELLS_CONCENTRATION = 0.01
    # secondary oxidants
    # OXIDANTS.SECONDARY.ELEMENT = "N"
    # OXIDANTS.SECONDARY.DIFFUSION_CONDITION = "N in Ni Krupp"
    # OXIDANTS.SECONDARY.CELLS_CONCENTRATION = 0.01
    # primary actives
    ACTIVES.PRIMARY.ELEMENT = "Cr"
    ACTIVES.PRIMARY.DIFFUSION_CONDITION = "Cr in Ni Krupp"
    ACTIVES.PRIMARY.MASS_CONCENTRATION = 0.11
    ACTIVES.PRIMARY.CELLS_CONCENTRATION = 0.2
    # secondary actives
    # ACTIVES.SECONDARY.ELEMENT = "None"
    # ACTIVES.SECONDARY.DIFFUSION_CONDITION = "Al in Ni Krupp"
    # ACTIVES.SECONDARY.MASS_CONCENTRATION = 0.025
    # ACTIVES.SECONDARY.CELLS_CONCENTRATION = 0.08
    # matrix
    MATRIX.ELEMENT = "Ni"

    TEMPERATURE = 1100  # °C
    N_CELLS_PER_AXIS = 102  # ONLY MULTIPLES OF 3+(neigh_range-1)*2 ARE ALLOWED
    N_ITERATIONS = 300000  # must be >= n_cells_per_axis
    STRIDE = 40  # n_iterations / stride = n_iterations for outward diffusion
    SIM_TIME = 72000  # [sek]
    SIZE = 500 * (10 ** -6)  # [m]
    SOL_PROD = 6.25 * 10 ** -31  # 5.621 * 10 ** -10
    PHASE_FRACTION_LIMIT = 0.045
    THRESHOLD_INWARD = 1
    THRESHOLD_OUTWARD = 1
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
    SAVE_POST_PROCESSED_INPUT = False
    # PROBABILITIES_______________________________________________________________
    PROBABILITIES = ElementGroups()
    PROBABILITIES.PRIMARY = ConfigProbabilities()
    # PROBABILITIES.SECONDARY = ConfigProbabilities()
    # PROBABILITIES.TERNARY = ConfigProbabilities()
    # PROBABILITIES.QUATERNARY = ConfigProbabilities()
    # nucleation primary___________________________
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
    # dissolution primary_________________________
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
    PROBABILITIES.PRIMARY.n = 2
    PROBABILITIES.PRIMARY.bsf = 1
    PROBABILITIES.PRIMARY.dissol_adapt_function = 3
    # ________________________
    GENERATED_VALUES = GeneratedValues()
    COMMENT = "NO COMMENTS"
