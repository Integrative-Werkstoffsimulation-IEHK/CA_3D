from engine import *
import traceback
from configuration import Config

if __name__ == '__main__':

    Config.COMMENT = """
    
    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale

    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_atomic_with_kinetic
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single

    eng.decomposition = eng.dissolution_atomic_with_kinetic
    eng.decomposition_intrinsic = eng.dissolution_zhou_wei_no_bsf_also_partial_neigh_aip

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
    eng.decomposition_intrinsic = eng.dissolution_zhou_wei_no_bsf_also_partial_neigh_aip

    eng.cur_case = eng.cases.first
    eng.cases.first.go_around_func_ref = eng.go_around_mult_oxid_n_also_partial_neigh_aip

    eng.cur_case.nucleation_probabilities = utils.NucleationProbabilities(Config.PROBABILITIES.PRIMARY,
                                                                          Config.PRODUCTS.PRIMARY)
    eng.cur_case.dissolution_probabilities = utils.DissolutionProbabilities(Config.PROBABILITIES.PRIMARY, Config.PRODUCTS.PRIMARY)

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
