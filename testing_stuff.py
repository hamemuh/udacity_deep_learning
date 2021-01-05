import numpy as np
mat1 = np.random.rand(2,1)
mat2 = np.random.rand(2,3)
# print(np.dot(mat2, mat1))
print("*" * 10)
# print(np.matmul(mat2, mat1))
print("*" * 10)
print(mat1.T.shape)
print(np.matmul(mat2.T, mat1))