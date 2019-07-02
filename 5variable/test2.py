import numpy as np

temp_input = np.array(
    [[8, 11],
     [11, 16],
     [88, 131],
     [37, 56],
     [55, 82],
     [199, 300],
     [141, 214],
     [125, 193],
     [133, 210],
     [169, 284],
     [127, 226]],
)

w = np.array([
    [2],
    [5],
])

b = np.array([7])

result = np.matmul(temp_input, w) + b
print(result)
