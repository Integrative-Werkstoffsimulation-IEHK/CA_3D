import traceback
from new_engine import *

if __name__ == '__main__':

    Config.COMMENT = """

    Nucleation and dissolution throughout the whole simulation (both schemes applied). Also with kinetic coefficient!!!
    Go along the kinetic growth line! Check the kinetic file as well!!!
"""
    new_system = SimulationConfigurator()
    new_system.configurate_functions()

    try:
        new_system.run_simulation()
    finally:
        try:

            if not Config.SAVE_WHOLE:
                # new_system.save_results()
                print()
        except (Exception,):
            # new_system.save_results()
            print("Not SAVED!")

        new_system.save_results()
        new_system.terminate_workers()

        # iterations = np.arange(new_system.ca.cumul_prod.last_in_buffer) * new_system.ca.precipitation_stride
        #
        # data = np.column_stack(
        #     (iterations, new_system.ca.cumul_prod.get_buffer(), new_system.ca.growth_rate.get_buffer()))
        # output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "_kinetics.txt"
        # with open(output_file_path, "w") as f:
        #     for row in data:
        #         f.write(" ".join(map(str, row)) + "\n")
        #
        # data = np.column_stack(
        #     (iterations, new_system.ca.cumul_prod1.get_buffer(), new_system.ca.growth_rate1.get_buffer()))
        # output_file_path = Config.SAVE_PATH + Config.GENERATED_VALUES.DB_ID + "1" + "_kinetics2.txt"
        # with open(output_file_path, "w") as f:
        #     for row in data:
        #         f.write(" ".join(map(str, row)) + "\n")

        new_system.insert_last_it()
        new_system.db.conn.commit()
        print()
        print("____________________________________________________________")
        print("Simulation was closed at Iteration: ", new_system.ca.iteration)
        print("____________________________________________________________")
        print()
        traceback.print_exc()
