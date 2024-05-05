from new_engine import *
import traceback
from configuration import Config

if __name__ == '__main__':

    Config.COMMENT = """ 
    
    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale
    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_atomic_opt_for_growth
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single
    eng.decomposition = eng.dissolution_test
    eng.cur_case = eng.cases.first
    In domain slides where product concentration reached the limit, nucleation and dissolution was stopped completely! 
"""

    eng = CellularAutomata()

    eng.primary_oxidant.diffuse = eng.primary_oxidant.diffuse_with_scale
    eng.primary_active.diffuse = eng.primary_active.diffuse_with_scale
    eng.precip_func = eng.precipitation_first_case
    eng.get_combi_ind = eng.get_combi_ind_atomic_opt_for_growth
    eng.precip_step = eng.precip_step_standard
    eng.check_intersection = eng.ci_single
    eng.decomposition = eng.dissolution_test
    eng.cur_case = eng.cases.first

    try:
        eng.simulation()
    finally:
        try:
            if not Config.SAVE_WHOLE:
                eng.save_results()

        except (Exception,):
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
        #     # data = np.column_stack(
        #     #     (np.arange(eng.iteration), eng.cumul_prod[:eng.iteration]))
        #     # output_file_path = "W:/SIMCA/test_runs_data/" + eng.utils.param["db_id"] + ".txt"
        #     # with open(output_file_path, "w") as f:
        #     #     for row in data:
        #     #         f.write(" ".join(map(str, row)) + "\n")

        eng.insert_last_it()
        eng.utils.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", eng.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
