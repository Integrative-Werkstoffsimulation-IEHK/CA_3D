import numpy as np

direction = [1, 3]
diff_jump_cells = 3

path_length = np.linalg.norm(direction)
print(path_length)

sing_ = direction / path_length
print(sing_)
print(np.linalg.norm(sing_))


vector_step = diff_jump_cells * sing_
print(vector_step)
print(np.linalg.norm(vector_step))


