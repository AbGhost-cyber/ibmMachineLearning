import numpy as np

list_a = [1, 2, 3]
list_b = ['a', 'b', 'c']

# for n, l in zip(list_a, list_b):
#     print(n)
#     print(l)
# for i in range(4):
#     print(i)
X = np.array([1, 2, 3, 4, 5])
reshape_x = X.reshape(-1, 1)
if __name__ == '__main__':
    print(X.shape)
    print(reshape_x.shape)

