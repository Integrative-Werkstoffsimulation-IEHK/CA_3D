# # go_around_bool signatures ______________________________
# @cc.export('go_around_bool_b1_i2', 'b1[:, :](b1[:, :, :], i2[:, :, :])')
# def go_around_bool_b1_i2(array_3d, arounds):
#     all_neighbours = []
#     single_neighbours = [np.ubyte(x) for x in range(0)]
#     for seed_arounds in arounds:
#         for point in seed_arounds:
#             single_neighbours.append(array_3d[point[0], point[1], point[2]])
#         all_neighbours.append(single_neighbours)
#         single_neighbours = [np.ubyte(x) for x in range(0)]
#     return np.array(all_neighbours, dtype=np.bool_)
#
#
# @cc.export('go_around_bool_u1_i2', 'b1[:, :](u1[:, :, :], i2[:, :, :])')
# def go_around_bool_u1_i2(array_3d, arounds):
#     all_neighbours = []
#     single_neighbours = [np.ubyte(x) for x in range(0)]
#     for seed_arounds in arounds:
#         for point in seed_arounds:
#             single_neighbours.append(array_3d[point[0], point[1], point[2]])
#         all_neighbours.append(single_neighbours)
#         single_neighbours = [np.ubyte(x) for x in range(0)]
#     return np.array(all_neighbours, dtype=np.bool_)
# # _______________________________________________________________
#
#
# # go_around_int signatures ______________________________
# @cc.export('go_around_int', 'u1[:, :](u1[:, :, :], i2[:, :, :])')
# def go_around_int(array_3d, arounds):
#     all_neighbours = []
#     single_neighbours = [np.ubyte(x) for x in range(0)]
#     for seed_arounds in arounds:
#         for point in seed_arounds:
#             single_neighbours.append(array_3d[point[0], point[1], point[2]])
#         all_neighbours.append(single_neighbours)
#         single_neighbours = [np.ubyte(x) for x in range(0)]
#     return np.array(all_neighbours, dtype=np.ubyte)
# # _______________________________________________________________
#
#
# # go_around_bool_dissol signatures ______________________________
# @cc.export('go_around_bool_dissol_b1_i2', 'b1[:, :](b1[:, :, :], i2[:, :, :])')
# def go_around_bool_dissol_b1_i2(array_3d, arounds):
#     all_neigh = []
#     single_neigh = [np.bool_(x) for x in range(0)]
#     for seed_arounds in arounds:
#         for point in seed_arounds:
#             single_neigh.append(array_3d[point[0], point[1], point[2]])
#         all_neigh.append(single_neigh)
#         single_neigh = [np.bool_(x) for x in range(0)]
#     return np.array(all_neigh, dtype=np.bool_)
#
#
# @cc.export('go_around_bool_dissol_u1_i2', 'b1[:, :](u1[:, :, :], i2[:, :, :])')
# def go_around_bool_dissol_u1_i2(array_3d, arounds):
#     all_neigh = []
#     single_neigh = [np.bool_(x) for x in range(0)]
#     for seed_arounds in arounds:
#         for point in seed_arounds:
#             single_neigh.append(array_3d[point[0], point[1], point[2]])
#         all_neigh.append(single_neigh)
#         single_neigh = [np.bool_(x) for x in range(0)]
#     return np.array(all_neigh, dtype=np.bool_)
# # _______________________________________________________________
#
#
# # check_at_coord_dissol signatures ______________________________
# @cc.export('check_at_coord_dissol_b1_i2', 'u4[:](b1[:, :, :], i2[:, :])')
# def check_at_coord_dissol_b1_i2(array_3d, coords):
#     result_coords = [np.uint32(x) for x in range(0)]
#     for coordinate in coords.transpose():
#         result_coords.append(array_3d[coordinate[0], coordinate[1], coordinate[2]])
#     return np.array(result_coords, dtype=np.uint32)
#
#
# @cc.export('check_at_coord_dissol_u1_i2(ubyte,short)', 'u4[:](u1[:, :, :], i2[:, :])')
# def check_at_coord_dissol_u1_i2(array_3d, coords):
#     # trick to initialize an empty list with known type
#     result_coords = [np.uint32(x) for x in range(0)]
#     for coordinate in coords.transpose():
#         result_coords.append(array_3d[coordinate[0], coordinate[1], coordinate[2]])
#     return np.array(result_coords, dtype=np.uint32)
# # _______________________________________________________________
#
#
# # check_at_coord signatures ______________________________
# @cc.export('check_at_coord_b1_i2', 'b1[:](b1[:, :, :], i2[:, :])')
# def check_at_coord_b1_i2(array_3d, coordinates):
#     result_coords = []
#     for i in range(coordinates.shape[1]):
#         x = coordinates[0, i]
#         y = coordinates[1, i]
#         z = coordinates[2, i]
#         result_coords.append(array_3d[x, y, z])
#     return np.array(result_coords, dtype=np.bool_)
#
#
# # @cc.export('check_at_coord_u1_i2', 'b1[:, :](u1[:, :, :], i2[:, :, :])')
# # def check_at_coord_u1_i2(array_3d, coordinates):
# #     # trick to initialize an empty list with known type
# #     result_coords = [np.bool_(x) for x in range(0)]
# #     for single_coordinate in coordinates:
# #         result_coords.append(array_3d[single_coordinate[0], single_coordinate[1], single_coordinate[2]])
# #     return np.array(result_coords, dtype=np.bool_)
#
# @cc.export('check_at_coord_u1_i2', 'u1[:](u1[:, :, :], i2[:, :])')
# def check_at_coord_u1_i2(array_3d, coordinates):
#     result_coords = []
#     for i in range(coordinates.shape[1]):
#         x = coordinates[0, i]
#         y = coordinates[1, i]
#         z = coordinates[2, i]
#         result_coords.append(array_3d[x, y, z])
#     return np.array(result_coords, dtype=np.ubyte)
#
# # _______________________________________________________________
#
#
# # insert_counts signatures ______________________________
# @cc.export('insert_counts', 'void(u1[:, :, :], i2[:, :])')
# def insert_counts(array_3d, points):
#     for point in points.transpose():
#         array_3d[point[0], point[1], point[2]] += 1
# # _______________________________________________________________
#
#
# # decrease_counts signatures ______________________________
# # @cc.export('decrease_counts', 'i8[:](u1[:, :, :], i2[:, :])')
# # def decrease_counts(array_3d, points):
# #     zero_positions = []
# #     for ind, point in enumerate(points.transpose()):
# #         if array_3d[point[0], point[1], point[2]] > 0:
# #             array_3d[point[0], point[1], point[2]] -= 1
# #         else:
# #             zero_positions.append(ind)
# #     return zero_positions
#
# @cc.export('decrease_counts', 'i8[:](u1[:, :, :], i2[:, :])')
# def decrease_counts(array_3d, points):
#     zero_positions = np.zeros(points.shape[1], dtype=np.int64)
#     count = 0
#     for ind in range(points.shape[1]):
#         x = points[0, ind]
#         y = points[1, ind]
#         z = points[2, ind]
#         if array_3d[x, y, z] > 0:
#             array_3d[x, y, z] -= 1
#         else:
#             zero_positions[count] = ind
#             count += 1
#     return zero_positions[:count]
# # _______________________________________________________________
#
#
# # decrease_counts signatures ______________________________
# @cc.export('just_decrease_counts', 'void(u1[:, :, :], i2[:, :])')
# def just_decrease_counts(array_3d, points):
#     for point in points.transpose():
#         array_3d[point[0], point[1], point[2]] -= 1
# # _______________________________________________________________
#
#
# # check_in_scale signatures ______________________________
# @cc.export('check_in_scale_b1_i2_i1', 'u4[:](b1[:, :, :], i2[:, :], i1[:, :])')
# def check_in_scale_b1_i2_i1(scale, cells, dirs):
#     # trick to initialize an empty list with known type
#     out_scale = [np.uint32(x) for x in range(0)]
#     for index, coordinate in enumerate(cells.transpose()):
#         if not scale[coordinate[0], coordinate[1], coordinate[2]]:
#             out_scale.append(np.uint32(index))
#         else:
#             dirs[:, index] *= -1
#     return np.array(out_scale, dtype=np.uint32)
#
#
# @cc.export('check_in_scale_u1_i2_i1', 'u4[:](u1[:, :, :], i2[:, :], i1[:, :])')
# def check_in_scale_u1_i2_i1(scale, cells, dirs):
#     # trick to initialize an empty list with known type
#     out_scale = [np.uint32(x) for x in range(0)]
#     for index, coordinate in enumerate(cells.transpose()):
#         if not scale[coordinate[0], coordinate[1], coordinate[2]]:
#             out_scale.append(np.uint32(index))
#         else:
#             dirs[:, index] *= -1
#     return np.array(out_scale, dtype=np.uint32)
# # _______________________________________________________________
#
#
# # separate_in_gb signatures ______________________________
# # @cc.export('separate_in_gb', '(u4[:], u4[:])(b1[:])')
# # def separate_in_gb(bool_arr):
# #     # trick to initialize an empty list with known type
# #     out_gb = [np.uint32(x) for x in range(0)]
# #     in_gb = [np.uint32(x) for x in range(0)]
# #     for index, bool_item in enumerate(bool_arr):
# #         if bool_item:
# #             in_gb.append(np.uint32(index))
# #         else:
# #             out_gb.append(np.uint32(index))
# #     return np.array(in_gb, dtype=np.uint32), np.array(out_gb, dtype=np.uint32)
#
# @cc.export('separate_in_gb', 'Tuple((u4[:], u4[:]))(b1[:])')
# def separate_in_gb(bool_arr):
#     out_gb = [np.uint32(x) for x in range(0)]
#     in_gb = [np.uint32(x) for x in range(0)]
#     for index, bool_item in enumerate(bool_arr):
#         if bool_item:
#             in_gb.append(np.uint32(index))
#         else:
#             out_gb.append(np.uint32(index))
#     return np.array(in_gb, dtype=np.uint32), np.array(out_gb, dtype=np.uint32)
# # _______________________________________________________________
#
#
# # aggregate signatures ______________________________
# @cc.export('aggregate', 'u4[:](i8[:, :], b1[:, :])')
# def aggregate(aggregated_ind, all_neigh_bool):
#     # trick to initialize an empty list with known type
#     where_blocks = [np.uint32(x) for x in range(0)]
#     for index, item in enumerate(all_neigh_bool):
#         for step in aggregated_ind:
#             if np.sum(item[step]) == 7:
#                 where_blocks.append(np.uint32(index))
#                 break
#     return np.array(where_blocks, dtype=np.uint32)
# # _______________________________________________________________
#
#
# # aggregate_and_count signatures ______________________________
# @cc.export('aggregate_and_count', 'u4[:](i8[:, :], b1[:, :])')
# def aggregate_and_count(aggregated_ind, all_neigh_bool):
#     # trick to initialize an empty list with known type
#     block_counts = [np.uint32(x) for x in range(0)]
#     for item in all_neigh_bool:
#         curr_count = 0
#         for step in aggregated_ind:
#             if np.sum(item[step]) == 7:
#                 curr_count += 1
#         block_counts.append(np.uint32(curr_count))
#     return np.array(block_counts, dtype=np.uint32)
# # _______________________________________________________________
