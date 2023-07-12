import numpy as np


class MyBufferCoords:
    def __init__(self, initial_shape, dtype=np.short):
        self.buffer = np.empty(initial_shape, dtype=dtype)
        self.last_in_buffer = 0

    def copy_to_buffer(self, data_array):
        self.buffer[:, :data_array.shape[1]] = data_array
        self.last_in_buffer = data_array.shape[1]

    def get_elements_at_indexes(self, indexes):
        return self.buffer[:, indexes].view()

    def get_elements_instead_indexes(self, indexes):
        return np.delete(self.buffer[:, :self.last_in_buffer], indexes, axis=1)

    def update_buffer_at_axis(self, new_data, axis=2):
        self.buffer[axis, :self.last_in_buffer] = new_data

    def get_buffer(self):
        return self.buffer[:, :self.last_in_buffer].view()


class MyBufferSingle:
    def __init__(self, initial_shape, dtype=np.uint32):
        self.buffer = np.empty(initial_shape, dtype=dtype)
        self.last_in_buffer = 0

    def copy_to_buffer(self, data_array):
        self.buffer[:data_array.shape[1]] = data_array
        self.last_in_buffer = data_array.shape[1]

    def update_buffer(self, new_data):
        self.buffer[:new_data.shape] = new_data
        self.last_in_buffer = new_data.shape

    def fill_buffer(self, furthest_index, array_3d):
        data = np.array(np.sum(array_3d[:, :, np.arange(furthest_index + 1)]), dtype=np.uint32)
        self.update_buffer(data)

    def get_buffer(self):
        return self.buffer[:self.last_in_buffer]
