import sqlite3 as sql
import time


class Database:
    def __init__(self, user_input):
        db_name = user_input["save_path"] + str(int(time.time())) + '.db'
        user_input["save_path"] = db_name
        self.conn = sql.connect(db_name)
        self.c = self.conn.cursor()
        self.save_user_input(user_input)
        self.save_elements_data(user_input)
        self.create_precipitation_front_table(user_input)
        self.create_time_parameters_table()

    def save_user_input(self, user_input):
        query = """CREATE TABLE user_input ("""
        for position, item in enumerate(user_input.items()):
            if position > 2:
                if str(type(item[1]))[8:-2] == "str":
                    query = query + item[0] + " " + "text, "
                elif str(type(item[1]))[8:-2] == "bool":
                    query = query + item[0] + " " + "int, "
                else:
                    query = query + item[0] + " " + str(type(item[1]))[8:-2] + ", "
        query = query[:-2] + ")"
        self.c.execute(query)

        query = """INSERT INTO user_input VALUES ("""
        for position, item in enumerate(user_input.items()):
            if position > 2:
                if str(type(item[1]))[8:-2] == "bool":
                    query = query + str(int(item[1])) + ", "
                elif str(type(item[1]))[8:-2] == "str":
                    query = query + f"""'{str(item[1])}'""" + ", "
                else:
                    query = query + str(item[1]) + ", "
        query = query[:-2] + ")"
        self.c.execute(query)

    def save_elements_data(self, user_input):
        # 0 - primary active element; 1 - secondary active element;
        # 2 - primary oxidant; 3 - secondary oxidant, 4 - matrix element
        query = f"""CREATE TABLE element_0 (elem text, diffusion_condition text, mass_concentration float,
        cells_concentration float)"""
        self.c.execute(query)
        elem = user_input["active_element"]["primary"]["elem"]
        condition = user_input["active_element"]["primary"]["diffusion_condition"]
        mass_concentration = user_input["active_element"]["primary"]["mass_concentration"]
        cells_concentration = user_input["active_element"]["primary"]["cells_concentration"]
        query = f"""INSERT INTO element_0 VALUES ('{str(elem)}', '{str(condition)}', {mass_concentration},
{cells_concentration})"""
        self.c.execute(query)

        query = f"""CREATE TABLE element_1 (elem text, diffusion_condition text, mass_concentration float,
        cells_concentration float)"""
        self.c.execute(query)
        elem = user_input["active_element"]["secondary"]["elem"]
        condition = user_input["active_element"]["secondary"]["diffusion_condition"]
        mass_concentration = user_input["active_element"]["secondary"]["mass_concentration"]
        cells_concentration = user_input["active_element"]["secondary"]["cells_concentration"]
        query = f"""INSERT INTO element_1 VALUES ('{str(elem)}', '{str(condition)}', {mass_concentration},
{cells_concentration})"""
        self.c.execute(query)

        query = f"""CREATE TABLE element_2 (elem text, diffusion_condition text, cells_concentration float)"""
        self.c.execute(query)
        elem = user_input["oxidant"]["primary"]["elem"]
        condition = user_input["oxidant"]["primary"]["diffusion_condition"]
        concentration = user_input["oxidant"]["primary"]["cells_concentration"]
        query = f"""INSERT INTO element_2 VALUES ('{str(elem)}', '{str(condition)}', {concentration})"""
        self.c.execute(query)

        query = f"""CREATE TABLE element_3 (elem text, diffusion_condition text, cells_concentration float)"""
        self.c.execute(query)
        elem = user_input["oxidant"]["secondary"]["elem"]
        condition = user_input["oxidant"]["secondary"]["diffusion_condition"]
        concentration = user_input["oxidant"]["secondary"]["cells_concentration"]
        query = f"""INSERT INTO element_3 VALUES ('{str(elem)}', '{str(condition)}', {concentration})"""
        self.c.execute(query)

        query = f"""CREATE TABLE element_4 (elem text, diffusion_condition text, concentration float)"""
        self.c.execute(query)
        elem = user_input["matrix_elem"]["elem"]
        condition = user_input["matrix_elem"]["diffusion_condition"]
        concentration = user_input["matrix_elem"]["concentration"]
        query = f"""INSERT INTO element_4 VALUES ('{str(elem)}', '{str(condition)}', {concentration})"""
        self.c.execute(query)

    def insert_particle_data(self, particle_type, iteration, data):
        """Particle types allowed: primary_oxidant, secondary_oxidant, primary_active, secondary_active,
        primary_product, secondary_product, ternary_product, quaternary_product"""

        query = """CREATE TABLE {}_iter_{} (z int, y int, x int)""".format(particle_type, str(iteration))
        self.c.execute(query)
        query = "INSERT INTO {}_iter_{} VALUES(?, ?, ?);".format(particle_type, str(iteration))
        data = data.transpose()
        data = self.to_tuple(data)
        self.c.executemany(query, data)

    def create_precipitation_front_table(self, user_input):
        self.c.execute(f"""CREATE TABLE precip_front_p (sqrt_time int, position int)""")
        if user_input["active_element"]["secondary"]["elem"] != "None":
            self.c.execute("""CREATE TABLE precip_front_s (sqrt_time int, position int)""")

    def insert_precipitation_front(self, sqrt_time, position, sign):
        """sign p for primary product
            sign s for secondary product"""
        self.c.execute("INSERT INTO precip_front_{} VALUES ({}, {})".format(sign, sqrt_time, position))

    def create_time_parameters_table(self):
        self.c.execute("""CREATE TABLE time_parameters (last_i int, elapsed_time float)""")

        query = """INSERT INTO time_parameters VALUES(?, ?);"""
        self.c.execute(query, (0, 0,))

    def insert_time(self, elapsed_time):
        self.c.execute("""UPDATE time_parameters set elapsed_time = {}""".format(elapsed_time))

    def insert_last_iteration(self, last_i):
        self.c.execute("""UPDATE time_parameters set last_i = {}""".format(last_i))

    @staticmethod
    def to_tuple(points):
        points = points.tolist()
        return [(point[0], point[1], point[2]) for point in points]

