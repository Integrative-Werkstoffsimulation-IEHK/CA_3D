import pyvoro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bresenham
import numpy as np
import time


class Microstructure:
    def __init__(self):
        self.grain_boundaries = None
        self.destinations = None
        self.lines = np.array([[[0,0,0], [0,0,0]]])
        self.lines_faces = np.array([[[0,0,0], [0,0,0]]])
        self.n_cells_per_axis = None
        self.divisor = 150
        self.diff_jump = 0.1
        self.diff_jump_cells = 2
        self.destinations = None
        self.ca_edges = None
        self.ca_faces = None

    def voronoi_3d_cells(self, n_cells_per_axis, number_of_grains, seeds=None):
        if seeds is None:
            seeds = np.random.random_sample((number_of_grains, 3))
        # regular arrangement of seeds for some tests
        elif seeds == 'regular8':
            seeds = [[(1 / 4), (1 / 4), (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4)],
                     [(1 / 4), (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4), (1 / 4), (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4)],
                     [(1 / 4) * 3, (1 / 4) * 3, (1 / 4) * 3],
                     [(1 / 4) * 3, (1 / 4), (1 / 4) * 3]]

        elif seeds == 'standard':
            seeds = np.array([[0.99546975, 0.18405673, 0.75009221],
                              [0.21039636, 0.22855524, 0.70656256],
                              [0.18267228, 0.75314614, 0.28313767],
                              [0.14943234, 0.3128683 , 0.28018282],
                              [0.42597253, 0.04042529, 0.99412492]])
        elif seeds == "plane":
            seeds = np.array([[0.1, 0.5, 0.5],
                              [0.9, 0.5, 0.5],])

        vor = pyvoro.compute_voronoi(seeds, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 0.01, radii=[1.3, 1.4])
        # for item in vor:
        #     for sub_item in item.items():
        #         print(sub_item)
        #         print()
        #     print("_______________________________________________________________________________________________________")
        # list of edges
        lines = []
        # list of faces
        lines_faces = []
        # list of indexes of already checked cells
        ind_done_cells = []
        for cell_ind, cell in enumerate(vor):
            for face in cell['faces']:
                # check if face of adjacent cell is already checked
                if face['adjacent_cell'] not in ind_done_cells and face['adjacent_cell'] >= 0:
                    # list of coordinates
                    face_vertices_coordinates = np.array([cell['vertices'][index] for index in face['vertices']])
                    if len(face_vertices_coordinates) == 3:
                        self.three(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 4:
                        self.four(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 5:
                        self.five(face_vertices_coordinates)
                    elif len(face_vertices_coordinates) == 6:
                        self.six(face_vertices_coordinates)
                    else:
                        print(f"Write Function for {len(face_vertices_coordinates)} vertices ")
                        self.all_forms(face_vertices_coordinates)
            ind_done_cells.append(cell_ind)

        # face_vertices_coordinates = np.array([[0.0, 0.25, 0.0],
        #                                       [1.0, 0.25, 0.0],
        #                                       [1.0, 0.8, 1.0],
        #                                       [0.0, 0.8, 1.0]])
        #
        # if len(face_vertices_coordinates) == 3:
        #     self.three(face_vertices_coordinates)
        # elif len(face_vertices_coordinates) == 4:
        #     self.four(face_vertices_coordinates)
        # elif len(face_vertices_coordinates) == 5:
        #     self.five(face_vertices_coordinates)
        # elif len(face_vertices_coordinates) == 6:
        #     self.six(face_vertices_coordinates)
        # else:
        #     print(f"Write Function for {len(face_vertices_coordinates)} vertices ")
        #     self.all_forms(face_vertices_coordinates)

        # working with edges
        # self.divide_vectors()

        starts_edges = np.array(np.array(self.lines)[:, 0] * (n_cells_per_axis - 1), dtype=np.short)
        ends_edges = np.array(np.array(self.lines)[:, 1] * (n_cells_per_axis - 1), dtype=np.short)
        # res_edges = [bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1) for start, finish in
        #              zip(starts_edges, ends_edges)]
        res_edges = np.array([bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish
                              in zip(starts_edges, ends_edges)], dtype=object)

        shift = 10
        bresenham_s_edges = res_edges + shift
        self.destinations = np.zeros((n_cells_per_axis + 2 * shift, n_cells_per_axis + 2 * shift,
                                 n_cells_per_axis + 2 * shift, 3), dtype=np.short)

        # for line in bresenham_s_edges:
        #     if len(line) > 0:
        #         # print(line)
        #         final_dest = line[-1] - shift
        #         for coordinate in line:
        #             self.destinations[coordinate[0], coordinate[1], coordinate[2]] = final_dest

        for line in bresenham_s_edges:
            if len(line) > 0:
                direction = line[-1] - line[0]
                path_length = np.linalg.norm(direction)
                vector_step = self.diff_jump_cells * direction / path_length
                vector_step = np.array(np.rint(vector_step), dtype=np.short)

                # final_dest = line[-1] - shift
                for coordinate in line:
                    self.destinations[coordinate[0], coordinate[1], coordinate[2]] = vector_step

        # bresenham_points_edges = np.array([point for array2 in res_edges for array1 in array2 for point in array1],
        # dtype=np.short)
        bresenham_points_s_edges = np.array([point for array2 in bresenham_s_edges for point in array2],
                                            dtype=np.short).transpose()

        # shift = 10
        # bresenham_points_shifted_edges = bresenham_points_edges + shift

        # ca_shifted_edges = np.zeros((n_cells_per_axis + 2 * shift, n_cells_per_axis + 2 * shift,
        #                              n_cells_per_axis + 2 * shift), dtype=int)
        ca_shifted_edges = np.full((n_cells_per_axis + 2 * shift, n_cells_per_axis + 2 * shift,
                                     n_cells_per_axis + 2 * shift), False, dtype=bool)
        ca_shifted_edges[bresenham_points_s_edges[0], bresenham_points_s_edges[1], bresenham_points_s_edges[2]] = True

        # for point in bresenham_points_shifted_edges:
        #     ca_shifted_edges[int(point[0]), int(point[1]), int(point[2])] = 1

        ca_edges = ca_shifted_edges[shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift)]

        # working with faces
        starts_faces = np.array(np.array(self.lines_faces)[:, 0] * (n_cells_per_axis - 1), dtype=np.short)
        ends_faces = np.array(np.array(self.lines_faces)[:, 1] * (n_cells_per_axis - 1), dtype=np.short)
        # res_faces = [bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1) for start, finish in
        #              zip(starts_faces, ends_faces)]
        res_faces = np.array([bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish
                              in zip(starts_faces, ends_faces)], dtype=object)
        shift = 10
        bresenham_s_faces = res_faces + shift

        # for line in bresenham_s_faces:
        #     if len(line) > 0:
        #         # print(line)
        #         final_dest = line[-1] - shift
        #         for coordinate in line:
        #             self.destinations[coordinate[0], coordinate[1], coordinate[2]] = final_dest

        for line in bresenham_s_faces:
            if len(line) > 0:
                direction = line[-1] - line[0]
                path_length = np.linalg.norm(direction)
                vector_step = self.diff_jump_cells * direction / path_length
                vector_step = np.array(np.rint(vector_step), dtype=np.short)

                # final_dest = line[-1] - shift
                for coordinate in line:
                    self.destinations[coordinate[0], coordinate[1], coordinate[2]] = vector_step

        bresenham_points_s_faces = np.array([point for array2 in bresenham_s_faces for point in array2]).transpose()
        # shift = 10
        # bresenham_points_shifted_faces = bresenham_points_faces + shift
        ca_shifted_faces = np.full((n_cells_per_axis + 2 * shift, n_cells_per_axis + 2 * shift,
                                    n_cells_per_axis + 2 * shift), False, dtype=bool)
        ca_shifted_faces[bresenham_points_s_faces[0], bresenham_points_s_faces[1], bresenham_points_s_faces[2]] = True

        # for point in bresenham_points_shifted_faces:
        #     ca_shifted_faces[int(point[0]), int(point[1]), int(point[2])] = 1
        ca_faces = ca_shifted_faces[shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift)]

        self.destinations = self.destinations[shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift),
                                    shift:(n_cells_per_axis + shift)]

        # edges
        # removing plane Z = 0
        ca_edges[0, :, :] = 0
        # removing plane Z = max
        ca_edges[n_cells_per_axis - 1, :, :] = 0
        # removing plane X = 0
        ca_edges[:, 0, :] = 0
        # removing plane X = max
        ca_edges[:, n_cells_per_axis - 1, :] = 0
        # removing plane Y = 0
        ca_edges[:, :, 0] = 0
        # removing plane Y = max
        ca_edges[:, :, n_cells_per_axis - 1] = 0

        # faces
        # removing plane Z = 0
        ca_faces[0, :, :] = 0
        # removing plane Z = max
        ca_faces[n_cells_per_axis - 1, :, :] = 0
        # removing plane X = 0
        ca_faces[:, 0, :] = 0
        # removing plane X = max
        ca_faces[:, n_cells_per_axis - 1, :] = 0
        # removing plane Y = 0
        ca_faces[:, :, 0] = 0
        # removing plane Y = max
        ca_faces[:, :, n_cells_per_axis - 1] = 0

        # destinations
        # removing plane Z = 0
        self.destinations[0, :, :] = [0, 0, 0]
        # removing plane Z = max
        self.destinations[n_cells_per_axis - 1, :, :] = [0, 0, 0]
        # removing plane X = 0
        self.destinations[:, 0, :] = [0, 0, 0]
        # removing plane X = max
        self.destinations[:, n_cells_per_axis - 1, :] = [0, 0, 0]
        # removing plane Y = 0
        self.destinations[:, :, 0] = [0, 0, 0]
        # removing plane Y = max
        self.destinations[:, :, n_cells_per_axis - 1] = [0, 0, 0]

        # return ca_faces
        self.ca_edges = np.nonzero(ca_edges)
        self.ca_faces = np.nonzero(ca_faces)

        shape = (n_cells_per_axis, n_cells_per_axis, n_cells_per_axis)
        self.grain_boundaries = np.full(shape, False, dtype=bool)

        self.grain_boundaries[self.ca_edges[0], self.ca_edges[1], self.ca_edges[2]] = True
        self.grain_boundaries[self.ca_faces[0], self.ca_faces[1], self.ca_faces[2]] = True
        # return whole_microstructure

        # return np.array(ca_faces), np.array(ca_edges)
        # return destinations
        return np.nonzero(ca_faces), np.nonzero(ca_edges)
        # return ca_faces

    def three(self, face_vert_coords):
        #  find the line closest to origin
        # print(face_vert_coords)
        min_x = np.min(face_vert_coords[:, 2])
        # print(min_x)
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        # print(min_ind)
        min_ind = min_ind[0]
        # print(min_ind)

        base_line_start = face_vert_coords[min_ind]

        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)
        # print(face_vert_coords)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            middle_point = face_vert_coords[1]
        else:
            base_line_end = face_vert_coords[1]
            middle_point = face_vert_coords[-1]

        # print("base_line_start ", base_line_start)
        # print("base_line_end ", base_line_end)
        # print("middle_point ", middle_point)

        base_line_m = (base_line_end - base_line_start) / 2
        # print(base_line_m)

        base_line_m = base_line_start + base_line_m
        # print(base_line_m)

        #  filling two areas
        first_area = self.fill_plane(base_line_start, base_line_start, base_line_m, middle_point)
        second_area = self.fill_plane(base_line_m, middle_point, base_line_end, base_line_end)

        # print(self.lines_faces)
        # self.lines_faces = np.concatenate((self.lines_faces, first_area), 0)
        self.lines_faces = np.append(self.lines_faces, first_area, axis=0)
        # print(self.lines_faces)
        self.lines_faces = np.append(self.lines_faces, second_area, axis=0)
        # print(self.lines_faces)

        #  base line vect
        # print("base line diff vector ", base_line_start, base_line_end)
        self.lines = np.append(self.lines, [[base_line_start, base_line_end]], axis=0)

        #  first line
        # print("first line vector ", base_line_start, middle_point)
        self.lines = np.append(self.lines, [[base_line_start, middle_point]], axis=0)

        #  second line
        # print("second line vector ", base_line_end, middle_point)
        self.lines = np.append(self.lines, [[base_line_end, middle_point]], axis=0)

    def four(self, face_vert_coords):
        #  find the line closest to origin
        # print(face_vert_coords)
        min_x = np.min(face_vert_coords[:, 2])
        # print(min_x)
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        # print(min_ind)
        min_ind = min_ind[0]
        # print(min_ind)

        base_line_start = face_vert_coords[min_ind]

        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)
        # print(face_vert_coords)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[1]
            pair_line_end = face_vert_coords[2]

        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[3]
            pair_line_end = face_vert_coords[2]

        # print("base_line_start ", base_line_start)
        # print("base_line_end ", base_line_end)
        # print("pair_line_start ", pair_line_start)
        # print("pair_line_end ", pair_line_end)

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)

        #  baseline vect
        # print("base line diff vector ", base_line_start, base_line_end)
        self.lines = np.append(self.lines, [[base_line_start, base_line_end]], axis=0)

        #  first line
        # print("first line vector ", base_line_start, pair_line_start)
        self.lines = np.append(self.lines, [[base_line_start, pair_line_start]], axis=0)

        #  second line
        # print("second line vector ", base_line_end, pair_line_end)
        self.lines = np.append(self.lines, [[base_line_end, pair_line_end]], axis=0)

        #  third line
        # print("third line vector ", pair_line_start, pair_line_end)
        self.lines = np.append(self.lines, [[pair_line_start, pair_line_end]], axis=0)

    def five(self, face_vert_coords):
        order = [[0, 1], [0, 4], [4, 3], [1, 2], [2, 3]]
        #  find the line closest to origin
        # print(face_vert_coords)
        min_x = np.min(face_vert_coords[:, 2])
        # print(min_x)
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        # print(min_ind)
        min_ind = min_ind[0]
        # print(min_ind)

        base_line_start = face_vert_coords[min_ind]

        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)
        # print(face_vert_coords)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[2]
            pair_line_end = face_vert_coords[3]

            tri_line_b_e = face_vert_coords[1]
            tri_line_p_s = face_vert_coords[2]

        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[3]
            pair_line_end = face_vert_coords[2]

            tri_line_b_e = face_vert_coords[-1]
            tri_line_p_s = face_vert_coords[3]

        # print("base_line_start ", base_line_start)
        # print("base_line_end ", base_line_end)
        # print("pair_line_start ", pair_line_start)
        # print("pair_line_end ", pair_line_end)

        # print("tri_line_b_e ", tri_line_b_e)
        # print("tri_line_p_s ", tri_line_p_s)

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)
        #  filling  triangle
        triangle = self.fill_plane(base_line_start, tri_line_p_s, tri_line_b_e, tri_line_b_e)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)

        edges_vectors = np.array([[face_vert_coords[pair[0]], face_vert_coords[pair[1]]] for pair in order])
        # print("""edges_vectors """)
        # print(edges_vectors)

        # print(self.lines)
        self.lines = np.append(self.lines, edges_vectors, axis=0)
        # print(self.lines)

    def six(self, face_vert_coords):
        order = [[0, 1], [0, 5], [5, 4], [4, 3], [1, 2], [2, 3]]
        #  find the line closest to origin
        # print(face_vert_coords)
        min_x = np.min(face_vert_coords[:, 2])
        # print(min_x)
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        # print(min_ind)
        min_ind = min_ind[0]
        # print(min_ind)

        base_line_start = face_vert_coords[min_ind]

        face_vert_coords = np.roll(face_vert_coords, -min_ind, axis=0)
        # print(face_vert_coords)

        if face_vert_coords[-1, 2] < face_vert_coords[1, 2]:
            base_line_end = face_vert_coords[-1]
            pair_line_start = face_vert_coords[2]
            pair_line_end = face_vert_coords[3]

            first_tri_p = face_vert_coords[1]
            sec_tri_p = face_vert_coords[4]

            first_tri_e = face_vert_coords[2]
            sec_tri_e = face_vert_coords[3]

        else:
            base_line_end = face_vert_coords[1]
            pair_line_start = face_vert_coords[4]
            pair_line_end = face_vert_coords[3]

            first_tri_p = face_vert_coords[5]
            sec_tri_p = face_vert_coords[2]

            first_tri_e = face_vert_coords[4]
            sec_tri_e = face_vert_coords[3]

        # print("base_line_start ", base_line_start)
        # print("base_line_end ", base_line_end)
        # print("pair_line_start ", pair_line_start)
        # print("pair_line_end ", pair_line_end)
        #
        # print("first_tri_p ", first_tri_p)
        # print("sec_tri_p ", sec_tri_p)
        #
        # print("first_tri_e ", first_tri_e)
        # print("sec_tri_e ", sec_tri_e)

        #  filling  area
        area = self.fill_plane(base_line_start, pair_line_start, base_line_end, pair_line_end)
        self.lines_faces = np.append(self.lines_faces, area, axis=0)

        #  filling first triangle
        triangle = self.fill_plane(base_line_start, first_tri_e, first_tri_p, first_tri_p)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)

        #  filling second triangle
        triangle = self.fill_plane(base_line_end, sec_tri_e, sec_tri_p, sec_tri_p)
        self.lines_faces = np.append(self.lines_faces, triangle, axis=0)

        edges_vectors = np.array([[face_vert_coords[pair[0]], face_vert_coords[pair[1]]] for pair in order])
        # print("""edges_vectors """)
        # print(edges_vectors)

        # print(self.lines)
        self.lines = np.append(self.lines, edges_vectors, axis=0)
        # print(self.lines)

    def all_forms(self, face_vert_coords):
        # face_vertices_coordinates = [cell['vertices'][index] for index in face['vertices']]
        # print(face_vertices_coordinates)
        # print(face_vertices_coordinates[:, 2])
        min_x = np.min(face_vert_coords[:, 2])
        # print(min_x)
        min_ind = np.array(np.where(face_vert_coords[:, 2] == min_x)[0])
        # print(min_ind)
        min_ind = min_ind[0]
        # print(min_ind)

        face_vertices_coordinates = np.roll(face_vert_coords, -min_ind, axis=0)
        # print(face_vertices_coordinates)

        zero_s = list(np.around(face_vertices_coordinates[0], decimals=3))
        # adding face's edges and some diagonals to divide it on triangles
        for i, coordinate in enumerate(face_vertices_coordinates):
            if i > 0:
                # edg
                current = list(np.around(coordinate, decimals=3))
                previous = list(np.around(face_vertices_coordinates[i - 1], decimals=3))
                # if [current, previous] not in self.lines and [previous, current] not in self.lines:
                    # lines.append([current, previous])
                self.lines = np.append(self.lines, [[current, previous]], axis=0)
                if i > 1:
                    # diagonal (or closing line in case of the last edge)
                    # if [current, zero_s] not in self.lines and [zero_s, current] not in self.lines:
                    if i == len(face_vertices_coordinates) - 1:
                        # lines.append([zero_s, current])
                        self.lines = np.append(self.lines, [[zero_s, current]], axis=0)
                    else:
                        # lines_faces.append([zero_s, current])
                        self.lines_faces = np.append(self.lines_faces, [[zero_s, current]], axis=0)
                    # filling triangle with lines
                    step = 1 / self.divisor
                    position = 0
                    path_start = np.array(current)
                    path_end = np.array(previous)
                    path_length = np.linalg.norm(path_end - path_start)
                    vector_step = (path_end - path_start) * step / path_length
                    while position < path_length:
                        path_start += vector_step
                        path_start_round = list(np.around(path_start, decimals=3))
                        position += step
                        # lines_faces.append([zero_s, path_start_round])
                        self.lines_faces = np.append(self.lines_faces, [[zero_s, path_start_round]], axis=0)

    def fill_plane(self, start_line_a, start_line_b, end_line_a, end_line_b):
        # divisor = 10
        starts = np.linspace(start_line_a, end_line_a, self.divisor)
        ends = np.linspace(start_line_b, end_line_b, self.divisor)
        new_lines_faces = np.concatenate((starts, ends), 1)
        new_lines_faces = np.reshape(new_lines_faces, (len(new_lines_faces), 2, 3))
        return new_lines_faces

    def divide_vectors(self):
        divided_lines = np.array([[[0, 0, 0], [0, 0, 0]]])
        for vector in self.lines:
            vector_length = np.linalg.norm(vector[1] - vector[0])
            n_of_steps = int(vector_length / self.diff_jump)
            if n_of_steps > 1:
                new_vectors = np.linspace(vector[0], vector[1], n_of_steps)
                new_vector_sec = np.array([[start_point, new_vectors[index+1]] for index, start_point in
                                           enumerate(new_vectors) if index < len(new_vectors) - 1])
                divided_lines = np.append(divided_lines, new_vector_sec, axis=0)
        self.lines = divided_lines

        divided_lines = np.array([[[0, 0, 0], [0, 0, 0]]])
        for vector in self.lines_faces:
            vector_length = np.linalg.norm(vector[1] - vector[0])
            n_of_steps = int(vector_length / self.diff_jump)
            if n_of_steps > 1:
                new_vectors = np.linspace(vector[0], vector[1], n_of_steps)
                new_vector_sec = np.array([[start_point, new_vectors[index+1]] for index, start_point in
                                           enumerate(new_vectors) if index < len(new_vectors) - 1])
                divided_lines = np.append(divided_lines, new_vector_sec, axis=0)
        self.lines_faces = divided_lines

    def show_microstructure(self, n_cells_per_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.ca_faces[2], self.ca_faces[1], self.ca_faces[0], color='darkgoldenrod', marker=',', s=1)
        ax.scatter(self.ca_edges[2], self.ca_edges[1], self.ca_edges[0], color='b', marker=',', s=5)
        ax.scatter(0, 50, 50, color='r', marker=',', s=50)

        ax.set_xlim3d(0, n_cells_per_axis)
        ax.set_ylim3d(0, n_cells_per_axis)
        ax.set_zlim3d(0, n_cells_per_axis)

        plt.show()


# # Some tests
if __name__ == "__main__":
    # # _______Plot 3D______
    # begin = time.time()
    # micro = Microstructure()
    # cells_faces, cells_edges = micro.voronoi_3d_cells(100, 50, seeds='regular8')
    # x_edge = cells_edges[2]
    # y_edge = cells_edges[1]
    # z_edge = cells_edges[0]
    # x_face = cells_faces[2]
    # y_face = cells_faces[1]
    # z_face = cells_faces[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_face, y_face, z_face, color='darkgoldenrod', marker=',', s=1)
    # ax.scatter(x_edge, y_edge, z_edge, color='b', marker=',', s=5)
    #
    # ax.scatter(0, 50, 50, color='r', marker=',', s=50)
    #
    # end = time.time()
    # elapsed_time = (end - begin)
    # print(f'elapsed time: {round(elapsed_time, 2)} s')
    # plt.show()

    # _____Plot slices_______
    begin = time.time()
    micro = Microstructure()
    cells_faces = micro.voronoi_3d_cells(100, 50)
    cut1 = np.nonzero(cells_faces[5, :, :])
    x1 = cut1[0]
    y1 = cut1[1]

    cut2 = np.nonzero(cells_faces[90, :, :])
    x2 = cut2[0]
    y2 = cut2[1]

    cut3 = np.nonzero(cells_faces[90, :, :])
    x3 = cut3[0]
    y3 = cut3[1]

    cut4 = np.nonzero(cells_faces[90, :, :])
    x4 = cut4[0]
    y4 = cut4[1]

    cut5 = np.nonzero(cells_faces[90, :, :])
    x5 = cut5[0]
    y5 = cut5[1]

    cut6 = np.nonzero(cells_faces[90, :, :])
    x6 = cut6[0]
    y6 = cut6[1]

    cut7 = np.nonzero(cells_faces[90, :, :])
    x7 = cut7[0]
    y7 = cut7[1]

    cut8 = np.nonzero(cells_faces[90, :, :])
    x8 = cut8[0]
    y8 = cut8[1]

    cut9 = np.nonzero(cells_faces[90, :, :])
    x9 = cut9[0]
    y9 = cut9[1]

    fig = plt.figure()
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    ax7 = fig.add_subplot(337)
    ax8 = fig.add_subplot(338)
    ax9 = fig.add_subplot(339)

    ax1.scatter(x1, y1, color='b', marker='s', s=10)
    ax2.scatter(x2, y2, color='b', marker='s', s=10)
    ax3.scatter(x3, y3, color='b', marker='s', s=10)
    ax4.scatter(x4, y4, color='b', marker='s', s=10)
    ax5.scatter(x5, y5, color='b', marker='s', s=10)
    ax6.scatter(x6, y6, color='b', marker='s', s=10)
    ax7.scatter(x7, y7, color='b', marker='s', s=10)
    ax8.scatter(x8, y8, color='b', marker='s', s=10)
    ax9.scatter(x9, y9, color='b', marker='s', s=10)

    end = time.time()
    elapsed_time = (end - begin)
    print(f'Elapsed time: {round(elapsed_time, 2)} s')
    plt.show()
