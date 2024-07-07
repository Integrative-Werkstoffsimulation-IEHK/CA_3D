from numba.pycc import CC
import numpy as np
import numba

cc = CC('precompiled_numba')


# go_around_bool signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('go_around_bool_b1_i2', 'b1[:, :](b1[:, :, :], i2[:, :, :])')
def go_around_bool_b1_i2(array_3d, arounds):
    all_neighbours = np.empty((arounds.shape[0], arounds.shape[1]), dtype=np.bool_)
    for i, seed_arounds in enumerate(arounds):
        for j, point in enumerate(seed_arounds):
            all_neighbours[i, j] = array_3d[point[0], point[1], point[2]]
    return all_neighbours


@numba.njit(nopython=True, fastmath=True)
@cc.export('go_around_bool_u1_i2', 'b1[:, :](u1[:, :, :], i2[:, :, :])')
def go_around_bool_u1_i2(array_3d, arounds):
    all_neighbours = np.empty((arounds.shape[0], arounds.shape[1]), dtype=np.bool_)
    for i, seed_arounds in enumerate(arounds):
        for j, point in enumerate(seed_arounds):
            all_neighbours[i, j] = array_3d[point[0], point[1], point[2]]
    return all_neighbours
# _______________________________________________________________


# go_around_int signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('go_around_int_c', 'u1[:, :](u1[:, :, :], i2[:, :, :])')
def go_around_int_c(array_3d, arounds):
    all_neighbours = np.empty((arounds.shape[0], arounds.shape[1]), dtype=np.ubyte)
    for i, seed_arounds in enumerate(arounds):
        for j, point in enumerate(seed_arounds):
            all_neighbours[i, j] = array_3d[point[0], point[1], point[2]]
    return all_neighbours
# _______________________________________________________________


# go_around_bool_dissol signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('go_around_bool_dissol_b1_i2', 'b1[:, :](b1[:, :, :], i2[:, :, :])')
def go_around_bool_dissol_b1_i2(array_3d, arounds):
    all_neigh = np.empty((arounds.shape[0], arounds.shape[1]), dtype=np.bool_)
    for i, seed_arounds in enumerate(arounds):
        for j, point in enumerate(seed_arounds):
            all_neigh[i, j] = array_3d[point[0], point[1], point[2]]
    return all_neigh


@numba.njit(nopython=True, fastmath=True)
@cc.export('go_around_bool_dissol_u1_i2', 'b1[:, :](u1[:, :, :], i2[:, :, :])')
def go_around_bool_dissol_u1_i2(array_3d, arounds):
    all_neigh = np.empty((arounds.shape[0], arounds.shape[1]), dtype=np.bool_)
    for i, seed_arounds in enumerate(arounds):
        for j, point in enumerate(seed_arounds):
            all_neigh[i, j] = array_3d[point[0], point[1], point[2]]
    return all_neigh
# _______________________________________________________________


# check_at_coord_dissol signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('check_at_coord_dissol_b1_i2', 'u4[:](b1[:, :, :], i2[:, :])')
def check_at_coord_dissol_b1_i2(array_3d, coords):
    result_coords = np.empty(coords.shape[1], dtype=np.uint32)
    for i, coordinate in enumerate(coords.T):
        result_coords[i] = array_3d[coordinate[0], coordinate[1], coordinate[2]]
    return result_coords

@numba.njit(nopython=True, fastmath=True)
@cc.export('check_at_coord_dissol_u1_i2', 'u4[:](u1[:, :, :], i2[:, :])')
def check_at_coord_dissol_u1_i2(array_3d, coords):
    result_coords = np.empty(coords.shape[1], dtype=np.uint32)
    for i, coordinate in enumerate(coords.T):
        result_coords[i] = array_3d[coordinate[0], coordinate[1], coordinate[2]]
    return result_coords
# _______________________________________________________________


# check_at_coord signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('check_at_coord_b1_i2', 'b1[:](b1[:, :, :], i2[:, :])')
def check_at_coord_b1_i2(array_3d, coordinates):
    result_coords = np.empty(coordinates.shape[1], dtype=np.bool_)
    for i in range(coordinates.shape[1]):
        x = coordinates[0, i]
        y = coordinates[1, i]
        z = coordinates[2, i]
        result_coords[i] = array_3d[x, y, z]
    return result_coords


@numba.njit(nopython=True, fastmath=True)
@cc.export('check_at_coord_u1_i2', 'u1[:](u1[:, :, :], i2[:, :])')
def check_at_coord_u1_i2(array_3d, coordinates):
    result_coords = np.empty(coordinates.shape[1], dtype=np.ubyte)
    for i in range(coordinates.shape[1]):
        x = coordinates[0, i]
        y = coordinates[1, i]
        z = coordinates[2, i]
        result_coords[i] = array_3d[x, y, z]
    return result_coords
# _______________________________________________________________


# insert_counts signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('insert_counts_c', 'void(u1[:, :, :], i2[:, :])')
def insert_counts_c(array_3d, points):
    for point in points.T:
        array_3d[point[0], point[1], point[2]] += 1
# _______________________________________________________________


# decrease_counts signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('decrease_counts_c', 'i8[:](u1[:, :, :], i2[:, :])')
def decrease_counts_c(array_3d, points):
    zero_positions = np.empty(points.shape[1], dtype=np.int64)
    count = 0
    for ind in range(points.shape[1]):
        x = points[0, ind]
        y = points[1, ind]
        z = points[2, ind]
        if array_3d[x, y, z] > 0:
            array_3d[x, y, z] -= 1
        else:
            zero_positions[count] = ind
            count += 1
    return zero_positions[:count]
# _______________________________________________________________


# decrease_counts signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('just_decrease_counts_c', 'void(u1[:, :, :], i2[:, :])')
def just_decrease_counts_c(array_3d, points):
    for point in points.T:
        array_3d[point[0], point[1], point[2]] -= 1
# _______________________________________________________________


# check_in_scale signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('check_in_scale_b1_i2_i1', 'u4[:](b1[:, :, :], i2[:, :], i1[:, :])')
def check_in_scale_b1_i2_i1(scale, cells, dirs):
    out_scale = np.empty(cells.shape[1], dtype=np.uint32)
    count = 0
    for index, coordinate in enumerate(cells.T):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale[count] = index
            count += 1
        else:
            dirs[:, index] *= -1
    return out_scale[:count]


@numba.njit(nopython=True, fastmath=True)
@cc.export('check_in_scale_u1_i2_i1', 'u4[:](u1[:, :, :], i2[:, :], i1[:, :])')
def check_in_scale_u1_i2_i1(scale, cells, dirs):
    out_scale = np.empty(cells.shape[1], dtype=np.uint32)
    count = 0
    for index, coordinate in enumerate(cells.T):
        if not scale[coordinate[0], coordinate[1], coordinate[2]]:
            out_scale[count] = index
            count += 1
        else:
            dirs[:, index] *= -1
    return out_scale[:count]
# _______________________________________________________________


# separate_in_gb signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('separate_in_gb_c', 'Tuple((u4[:], u4[:]))(b1[:])')
def separate_in_gb_c(bool_arr):
    in_gb = np.empty(len(bool_arr), dtype=np.uint32)
    out_gb = np.empty(len(bool_arr), dtype=np.uint32)
    in_count = 0
    out_count = 0
    for index, bool_item in enumerate(bool_arr):
        if bool_item:
            in_gb[in_count] = index
            in_count += 1
        else:
            out_gb[out_count] = index
            out_count += 1
    return in_gb[:in_count], out_gb[:out_count]
# _______________________________________________________________


# aggregate signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('aggregate_c', 'u4[:](i8[:, :], b1[:, :])')
def aggregate_c(aggregated_ind, all_neigh_bool):
    where_blocks = np.empty(all_neigh_bool.shape[0], dtype=np.uint32)
    count = 0
    for index, item in enumerate(all_neigh_bool):
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                where_blocks[count] = index
                count += 1
                break
    return where_blocks[:count]
# _______________________________________________________________


# aggregate_and_count signatures ______________________________
@numba.njit(nopython=True, fastmath=True)
@cc.export('aggregate_and_count_c', 'u4[:](i8[:, :], b1[:, :])')
def aggregate_and_count_c(aggregated_ind, all_neigh_bool):
    block_counts = np.empty(all_neigh_bool.shape[0], dtype=np.uint32)
    for index, item in enumerate(all_neigh_bool):
        curr_count = 0
        for step in aggregated_ind:
            if np.sum(item[step]) == 7:
                curr_count += 1
        block_counts[index] = curr_count
    return block_counts
# _______________________________________________________________


if __name__ == "__main__":
    cc.compile()
