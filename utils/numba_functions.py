import numba
import numpy as np


@numba.njit(nopython=True)
def go_around(array_3d, arrounds):
    all_neighbours = []
    # trick to initialize an empty list with known type
    single_neighbours = [np.ubyte(x) for x in range(0)]
    for seed_arrounds in arrounds:
        for point in seed_arrounds:
            single_neighbours.append(array_3d[point[0], point[1], point[2]])
        all_neighbours.append(single_neighbours)
        single_neighbours = [np.ubyte(x) for x in range(0)]
    return np.array(all_neighbours, dtype=np.bool_)


@numba.njit(nopython=True)
def check_at_coord(array_3d, coordinates):
    # trick to initialize an empty list with known type
    result_coords = [np.bool_(x) for x in range(0)]
    for single_coordinate in coordinates:
        result_coords.append(array_3d[single_coordinate[0], single_coordinate[1], single_coordinate[2]])
    return np.array(result_coords, dtype=np.bool_)


@numba.njit(nopython=True)
def insert_counts(array_3d, points):
    for point in points.transpose():
        array_3d[point[0], point[1], point[2]] += 1


@numba.njit(nopython=True)
def decrease_counts(array_3d, points):
    zero_positions = []
    for ind, point in enumerate(points.transpose()):
        if array_3d[point[0], point[1], point[2]] > 0:
            array_3d[point[0], point[1], point[2]] -= 1
        else:
            zero_positions.append(ind)
    return zero_positions


@numba.njit(nopython=True)
def check_in_scale(scale, coordinates):
    # trick to initialize an empty list with known type
    in_scale = [np.uint32(x) for x in range(0)]
    out_scale = [np.uint32(x) for x in range(0)]
    for index, single_coordinate in enumerate(coordinates.transpose()):
        if scale[single_coordinate[0], single_coordinate[1], single_coordinate[2]] > 0:
            in_scale.append(np.uint32(index))
        else:
            out_scale.append(np.uint32(index))
    return np.array(in_scale, dtype=np.uint32), np.array(out_scale, dtype=np.uint32)


@numba.njit(nopython=True)
def diff_single(directions, probs, random_numbs):
    for index, direction in enumerate(directions.transpose()):
        rand_numb = random_numbs.random()
        if rand_numb > probs[4]:
            new_direction = [direction[0], direction[1], direction[2]]
        elif probs[3] < rand_numb <= probs[4]:
            new_direction = [np.byte(direction[0] * -1), np.byte(direction[1] * -1), np.byte(direction[2] * -1)]
        elif rand_numb <= probs[0]:
            new_direction = [direction[2], direction[0], direction[1]]
        elif probs[0] < rand_numb <= probs[1]:
            new_direction = [np.byte(direction[2] * -1), np.byte(direction[0] * -1), np.byte(direction[1] * -1)]
        elif probs[1] < rand_numb <= probs[2]:
            new_direction = [direction[1], direction[2], direction[0]]
        else:
            new_direction = [np.byte(direction[1] * -1), np.byte(direction[2] * -1), np.byte(direction[0] * -1)]
        directions[:, index] = new_direction


@numba.njit(nopython=True)
def complete_diff_step(cells, directions, probs, random_numbs):
    for index, direction in enumerate(directions.transpose()):
        rand_numb = random_numbs.random()
        if rand_numb > probs[4]:
            new_direction = [direction[0], direction[1], direction[2]]
        elif probs[3] < rand_numb <= probs[4]:
            new_direction = [np.byte(direction[0] * -1), np.byte(direction[1] * -1), np.byte(direction[2] * -1)]
        elif rand_numb <= probs[0]:
            new_direction = [direction[2], direction[0], direction[1]]
        elif probs[0] < rand_numb <= probs[1]:
            new_direction = [np.byte(direction[2] * -1), np.byte(direction[0] * -1), np.byte(direction[1] * -1)]
        elif probs[1] < rand_numb <= probs[2]:
            new_direction = [direction[1], direction[2], direction[0]]
        else:
            new_direction = [np.byte(direction[1] * -1), np.byte(direction[2] * -1), np.byte(direction[0] * -1)]

        directions[:, index] = new_direction

        new_direction = [np.short(direction[0]), np.short(direction[1]), np.short(direction[2])]
        coord = cells[:, index]
        new_coord = [a + b for a, b in zip(new_direction, coord)]
        cells[:, index] = new_coord

        if cells[2, index] < 0:
            cells[2, index] = 1
            directions[2, index] = 1

        elif cells[0, index] == -1:
            cells[0, index] = 500

        elif cells[0, index] == 501:
            cells[0, index] = 0

        elif cells[1, index] == -1:
            cells[1, index] = 500

        elif cells[1, index] == 501:
            cells[1, index] = 0

        elif cells[2, index] == 501:
            cells[2, index] = 499
            directions[2, index] = -1



