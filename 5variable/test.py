import numpy as np

temp_input =np.random.randint(0,100,(200,2))


w = np.array([
    [1, 2],
    [2, 3],
])

b = np.array([3])

result = np.matmul(temp_input, w) + b
print(result)

