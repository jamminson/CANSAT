import numpy as np
import time
import cv2


# start = time.time()


def test1():
    pass
    # print(arr)
    # d_arr = 2 * arr
    # print(d_arr)
    #
    # # print(arr[:, 1] > 3)
    # arr1 = np.array([0, 0, 0])
    # arr2 = np.array([1, 1, 1])
    # arr3 = np.array([0, 1, 2])
    #
    # print(arr2 > arr1)
    #
    # if np.all(arr2 > arr1):
    #     arr3 = [0]
    #
    # print(arr3)
    #
    #
    # print(arr[0:4, 0:3])
    # [[1 1 1]
    #  [1 1 1]
    #  [1 1 1]
    #  [1 1 1]]


def test2():
    pass
    # arr = np.arange(16).reshape(4, 4)
    # new_arr = np.array_split(arr, 2, axis=0)
    # for array in new_arr:
    #     final_split = np.array_split(array, 3, axis=1)
    #     print(type(final_split))
    #
    # print(a[0, 0])

    # a = np.arange(3).reshape(1, 3)  # [0, 1, 2]
    # b = np.ones((1, 3))  # [1, 1, 1]
    # c = np.array([2, 2, 5], int)
    # d = np.array([3, 3, 3], int)
    #
    # if np.all(d > c):
    #     print('hi')

    # def second_iteration(no_along_x, no_along_y):
    #     # assume dimensions are divisible by the inputs respectively
    #     multiple_x = int(2 / no_along_x)
    #     multiple_y = int(4 / no_along_y)
    #
    #     for x_subset in range(multiple_x):
    #         starting_x_index = x_subset * no_along_x
    #         last_x_index = starting_x_index + no_along_x
    #
    #         for y_subset in range(multiple_y):
    #             starting_y_index = y_subset * no_along_y
    #             last_y_index = starting_y_index + no_along_y
    #             # Strictly speaking the last indexes and last indexes + 1 because the slicing will take down the index
    #             # by 1.
    #             print(starting_x_index, last_x_index)
    #             print(starting_y_index, last_y_index)
    #
    #             image_subset = arr[starting_x_index:last_x_index, starting_y_index:last_y_index]
    #             # image_subset = arr[0:2, 2:4]
    #             print(image_subset)
    #             print('Done one')

    # second_iteration(2, 2)


# x = np.arange(16).reshape(4, 4)
#
# # a = x[0:2, 2:4]
# # print(a)
# image_subsets = []
# no_subsets_x = 1
# no_subsets_y = 2
#
# # image_approx = np.zeros((no_subsets_x, no_subsets_y), int)
# # new_arr = np.array_split(x, no_subsets_x, axis=0)  # by row
# # for array in new_arr:
# #     image_subsets.append(np.array_split(array, no_subsets_y, axis=1))  # by column
# #     # This forms a list with each index being the representing a horizontal slice across, with elements among
# #     # the list its corresponding vertical slice.
# # print(image_subsets)
# # print(type(image_subsets[0][1]))
# # for row in range(no_subsets_x):
# #     for column in range(no_subsets_y):
# #         print(image_subsets[row][column])
#
# a = np.zeros((4, 4, 2), int)
# b = np.arange(32).reshape(4, 4, 2)
# print(b)
# print(b[0,:,:])


# end = time.time()
# elapsed_time = end - start
# print(elapsed_time)

image = cv2.imread('../Images/Test 3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original image', image)
cv2.waitKey(0)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(gray.shape)
