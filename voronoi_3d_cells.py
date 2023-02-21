import pyvoro
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bresenham
import numpy as np
import time


def voronoi_3d_cells(n_cells_per_axis, number_of_grains, seeds=None):
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
                # face_vertices_coordinates = [cell['vertices'][index] for index in face['vertices']]
                # print(face_vertices_coordinates)
                # print(face_vertices_coordinates[:, 2])
                min_x = np.min(face_vertices_coordinates[:, 2])
                # print(min_x)
                min_ind = np.array(np.where(face_vertices_coordinates[:, 2] == min_x)[0])
                # print(min_ind)
                min_ind = min_ind[0]
                # print(min_ind)

                face_vertices_coordinates = np.roll(face_vertices_coordinates, -min_ind, axis=0)
                # print(face_vertices_coordinates)

                zero_s = list(np.around(face_vertices_coordinates[0], decimals=3))
                # adding face's edges and some diagonals to divide it on triangles
                for i, coordinate in enumerate(face_vertices_coordinates):
                    if i > 0:
                        # edg
                        current = list(np.around(coordinate, decimals=3))
                        previous = list(np.around(face_vertices_coordinates[i - 1], decimals=3))
                        if [current, previous] not in lines and [previous, current] not in lines:
                            lines.append([current, previous])
                        if i > 1:
                            # diagonal (or closing line in case of the last edge)
                            if [current, zero_s] not in lines and [zero_s, current] not in lines:
                                if i == len(face_vertices_coordinates) - 1:
                                    lines.append([current, zero_s])
                                else:
                                    lines_faces.append([current, zero_s])
                        # filling triangle with lines
                        step = 1 / n_cells_per_axis
                        position = 0
                        path_start = np.array(current)
                        path_end = np.array(previous)
                        path_length = np.linalg.norm(path_end - path_start)
                        vector_step = (path_end - path_start) * step / path_length
                        while position < path_length:
                            path_start += vector_step
                            path_start_round = list(np.around(path_start, decimals=3))
                            position += step
                            lines_faces.append([zero_s, path_start_round])
        ind_done_cells.append(cell_ind)
    # working with edges
    starts_edges = np.array(np.array(lines)[:, 0] * (n_cells_per_axis - 1), dtype=np.short)
    ends_edges = np.array(np.array(lines)[:, 1] * (n_cells_per_axis - 1), dtype=np.short)
    # res_edges = [bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1) for start, finish in
    #              zip(starts_edges, ends_edges)]
    res_edges = np.array([bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish in
                 zip(starts_edges, ends_edges)], dtype=object)

    shift = 10
    bresenham_s_edges = res_edges + shift
    destinations = np.zeros((n_cells_per_axis + 2 * shift, n_cells_per_axis + 2 * shift,
                             n_cells_per_axis + 2 * shift, 3), dtype=np.short)

    for line in bresenham_s_edges:
        if len(line) > 0:
            # print(line)
            final_dest = line[-1]
            for coordinate in line:
                destinations[coordinate[0], coordinate[1], coordinate[2]] = final_dest


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
    starts_faces = np.array(np.array(lines_faces)[:, 0] * (n_cells_per_axis - 1), dtype=np.short)
    ends_faces = np.array(np.array(lines_faces)[:, 1] * (n_cells_per_axis - 1), dtype=np.short)
    # res_faces = [bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1) for start, finish in
    #              zip(starts_faces, ends_faces)]
    res_faces = np.array([bresenham.bresenhamlines(np.array([start]), np.array([finish]), -1)[0] for start, finish in
                          zip(starts_faces, ends_faces)], dtype=object)
    shift = 10
    bresenham_s_faces = res_faces + shift

    for line in bresenham_s_faces:
        if len(line) > 0:
            # print(line)
            final_dest = line[-1]
            for coordinate in line:
                destinations[coordinate[0], coordinate[1], coordinate[2]] = final_dest

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

    destinations = destinations[shift:(n_cells_per_axis + shift),
                                shift:(n_cells_per_axis + shift),
                                shift:(n_cells_per_axis + shift)]

    # # edges
    # # removing plane Z = 0
    # ca_edges[0, :, :] = 0
    # # removing plane Z = max
    # ca_edges[n_cells_per_axis - 1, :, :] = 0
    # # removing plane X = 0
    # ca_edges[:, 0, :] = 0
    # # removing plane X = max
    # ca_edges[:, n_cells_per_axis - 1, :] = 0
    # # removing plane Y = 0
    # ca_edges[:, :, 0] = 0
    # # removing plane Y = max
    # ca_edges[:, :, n_cells_per_axis - 1] = 0

    # cube edges
    ca_edges[0, 0, :] = 0
    ca_edges[0, :, 0] = 0
    ca_edges[:, 0, 0] = 0
    ca_edges[:, -1, -1] = 0
    ca_edges[-1, :, -1] = 0
    ca_edges[-1, -1, :] = 0
    ca_edges[:, 0, -1] = 0
    ca_edges[0, :, -1] = 0
    ca_edges[-1, 0, :] = 0
    ca_edges[:, -1, 0] = 0
    ca_edges[-1, :, 0] = 0
    ca_edges[0, -1, :] = 0

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
    # return ca_faces
    # ca_edges = np.nonzero(ca_edges)
    # ca_faces = np.nonzero(ca_faces)
    #
    # shape = (n_cells_per_axis, n_cells_per_axis, n_cells_per_axis)
    # whole_microstructure = np.full(shape, False, dtype=bool)
    #
    # whole_microstructure[ca_edges[0], ca_edges[1], ca_edges[2]] = True
    # whole_microstructure[ca_faces[0], ca_faces[1], ca_faces[2]] = True
    # return whole_microstructure

    # return np.array(ca_faces), np.array(ca_edges)
    # return destinations
    return np.nonzero(ca_faces), np.nonzero(ca_edges)


# Some tests
if __name__ == "__main__":
    # _______Plot 3D______
    begin = time.time()
    cells_faces, cells_edges = voronoi_3d_cells(100, 20)
    x_edge = cells_edges[1]
    y_edge = cells_edges[2]
    z_edge = cells_edges[0]
    x_face = cells_faces[1]
    y_face = cells_faces[2]
    z_face = cells_faces[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_face, y_face, z_face, color='darkgoldenrod', marker=',', s=1)
    ax.scatter(x_edge, y_edge, z_edge, color='b', marker=',', s=5)
    end = time.time()
    elapsed_time = (end - begin)
    print(f'elapsed time: {round(elapsed_time, 2)} s')
    plt.show()

    # _____Plot slices_______
    # begin = time.time()
    # cells_faces = voronoi_3d_cells(100, 5)[0]
    # cut1 = np.nonzero(cells_faces[5, :, :])
    # x1 = cut1[0]
    # y1 = cut1[1]
    #
    # cut2 = np.nonzero(cells_faces[90, :, :])
    # x2 = cut2[0]
    # y2 = cut2[1]
    #
    # cut3 = np.nonzero(cells_faces[90, :, :])
    # x3 = cut3[0]
    # y3 = cut3[1]
    #
    # cut4 = np.nonzero(cells_faces[90, :, :])
    # x4 = cut4[0]
    # y4 = cut4[1]
    #
    # cut5 = np.nonzero(cells_faces[90, :, :])
    # x5 = cut5[0]
    # y5 = cut5[1]
    #
    # cut6 = np.nonzero(cells_faces[90, :, :])
    # x6 = cut6[0]
    # y6 = cut6[1]
    #
    # cut7 = np.nonzero(cells_faces[90, :, :])
    # x7 = cut7[0]
    # y7 = cut7[1]
    #
    # cut8 = np.nonzero(cells_faces[90, :, :])
    # x8 = cut8[0]
    # y8 = cut8[1]
    #
    # cut9 = np.nonzero(cells_faces[90, :, :])
    # x9 = cut9[0]
    # y9 = cut9[1]
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(331)
    # ax2 = fig.add_subplot(332)
    # ax3 = fig.add_subplot(333)
    # ax4 = fig.add_subplot(334)
    # ax5 = fig.add_subplot(335)
    # ax6 = fig.add_subplot(336)
    # ax7 = fig.add_subplot(337)
    # ax8 = fig.add_subplot(338)
    # ax9 = fig.add_subplot(339)
    #
    # ax1.scatter(x1, y1, color='b', marker='s', s=10)
    # ax2.scatter(x2, y2, color='b', marker='s', s=10)
    # ax3.scatter(x3, y3, color='b', marker='s', s=10)
    # ax4.scatter(x4, y4, color='b', marker='s', s=10)
    # ax5.scatter(x5, y5, color='b', marker='s', s=10)
    # ax6.scatter(x6, y6, color='b', marker='s', s=10)
    # ax7.scatter(x7, y7, color='b', marker='s', s=10)
    # ax8.scatter(x8, y8, color='b', marker='s', s=10)
    # ax9.scatter(x9, y9, color='b', marker='s', s=10)
    #
    # end = time.time()
    # elapsed_time = (end - begin)
    # print(f'Elapsed time: {round(elapsed_time, 2)} s')
    # plt.show()
