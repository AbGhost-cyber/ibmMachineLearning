import numpy as np

array_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
reshaped_array = array_a.T
array_b = array_a.reshape(2, 8)
array_c = array_a.T
# print(array_a[:, -3])
# print(array_a[2, 1])
array_e = np.array([[1, 2, 3], [4, 5, 6]])
array_f = np.ones((5, 3))
array_g = np.full((3, 3), 5)
array_h = np.random.random((3, 3))

# slice
# print(array_a[:, 1:3:1])
# negative index
# print(array_a[:, -2: -4: -1])

# boolean
greater_than_ten = array_a > 10
true_values = array_a[greater_than_ten]
false_values = array_a[~greater_than_ten]
# print("true values: ", true_values)
# print("false values: ", false_values)

# where
drop_under_10_array = np.where(array_a > 10, array_a, 0)

# logical_and
drop_under_5_and_over_10 = np.logical_and(array_a > 5, array_a < 10)

# using a range
array_1 = np.arange(0, 100, 5)
array_1_reshape = array_1.reshape(4, 5)
array_1_above_50 = array_1_reshape[array_1_reshape > 50]
array_reverse_column = array_1_reshape[:, -1: -4: -1]

array2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array3 = np.array([[2, 2, 2], [3, 4, 5], [6, 7, 8]])

# single array operations, sum axis 1 = row, axis 0 = column
# print(array2.sum(axis=1))
# print(array2.cumsum())
# print(array2.prod())
# print(array2.cumprod())

# two array op can perform math op
# print(array2 + array3)
# print(array2.dot(array3))
# print(array2.ptp())
# print(array2.mean())
# print(np.power(array2, 2))

# broadcasting
array4 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
array5 = np.array([0, 1, 2, 3])
# print(array5 + array4)
array6 = np.ones((5, 5))
array7 = np.array([0, 1, 2, 3, 4])
print(array6 * array7)

#
if __name__ == '__main__':
    print()
