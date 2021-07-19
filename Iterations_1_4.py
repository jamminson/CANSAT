import numpy as np
import cv2
import time
from statistics import mean
import time


class Mask:

    def __init__(self, boundaries, image_name):

        self.boundaries = boundaries
        # The numpy arrays are the various boundaries for that the mask uses.
        # Note the order is BGR in these arrays.
        # (lower, upper)
        # boundary = boundaries = [[50, 0, 0], [220, 110, 110]]

        self.image_name = image_name
        image = cv2.imread(self.image_name)
        self.image_data = np.asarray(image)
        self.image_subsets = list()
        self.image_approx = None
        [self.shape_x, self.shape_y, self.shape_z] = self.image_data.shape
        # x is the rows, y the columns and z is the depth.

    def first_iteration(self):

        # for i in range(len(self.image_data)):
        #     for j in range(len(self.image_data[0])):
        #         if np.all(self.image_data[i][j] > self.boundaries[0]) & np.all(self.image_data[i][j] <
        #                                                                        self.boundaries[1]):
        #             self.image_data[i][j] = [0]
        #         else:
        #             self.image_data[i][j] = [1]

        # Second attempt

        for i in range(self.shape_x):
            for j in range(self.shape_y):
                if np.all(self.image_data[i, j] > self.boundaries[0]) & np.all(self.image_data[i, j] <
                                                                               self.boundaries[1]):
                    self.image_data[i, j] = [0]
                else:
                    self.image_data[i, j] = [1]

        # compare_pixels(self.image_data, self.boundaries, i, j)

    def second_iteration(self, no_subsets_x, no_subsets_y):

        # # assume dimensions are divisible by the inputs respectively
        # multiple_x = self.shape_x / no_along_x
        # multiple_y = self.shape_y / no_along_y
        #
        # for x_subset in range(multiple_x):
        #     for y_subset in range(multiple_y):
        #         starting_x_index = x_subset * no_along_x
        #         last_x_index = starting_x_index + no_along_x
        #         starting_y_index = y_subset * no_along_y
        #         last_y_index = starting_y_index + no_along_y
        #         # Strictly speaking the last indexes and last indexes + 1 because the slicing will take down the index
        #         # by 1.
        #
        #         image_subset = self.image_data[starting_x_index:last_x_index, starting_y_index:last_y_index]

        self.image_approx = np.zeros((no_subsets_x, no_subsets_y), int)
        new_arr = np.array_split(self.image_data, no_subsets_x, axis=0)  # by row
        for array in new_arr:
            self.image_subsets.append(np.array_split(array, no_subsets_y, axis=1))  # by column
            # This forms a list with each index being the representing a horizontal slice across, with elements among
            # the list its corresponding vertical slice.

        for row in range(no_subsets_x):
            for column in range(no_subsets_y):
                current_subset = self.image_subsets[row][column]
                current_subset_avg = average_value_subset(current_subset)

                # Fills in self.image_approx with whether each subset passed or not.
                compare_pixels(current_subset_avg, self.image_approx, self.boundaries, row, column)

        # Confirmation that there are parts where the condition fails
        # x = np.where(self.image_approx == 0)
        # print(x)

    def find_direction(self, pointer_length):
        last_starting_x = self.shape_x - pointer_length + 1
        last_starting_y = self.shape_y - pointer_length + 1

        for i in range(last_starting_x):
            # To include the last element, the index is the actual dimension.

            for j in range(last_starting_y):
                image_considered = self.image_approx[i:(i + pointer_length), j:(j + pointer_length)]

                print(image_considered)

    def third_iteration(self, kernel):
        blue_layer = self.image_data[:, :, 0]
        print(blue_layer.shape)
        (iH, iW) = blue_layer.shape[:2]
        (kH, kW) = kernel.shape[:2]
        print(iH)
        print(kH)
        pad = (kW - 1) // 2
        padded_image = cv2.copyMakeBorder(blue_layer, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        iteration_output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                roi = padded_image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                k = (roi * kernel).sum()
                iteration_output[y - pad, x - pad] = int(k / 9)

        return iteration_output

    def fourth_iteration(self):
        image = self.process()
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        output = self.convolve(image, kernel, padding=0, strides=5)
        cv2.imwrite('Images/2DConvolved.jpg', output)

    def process(self):
        blue_layer = self.image_data
        image = cv2.cvtColor(blue_layer, code=cv2.COLOR_BGR2GRAY)
        return image

    def convolve(self, image, kernel, padding=0, strides=1):
        kernel = np.flipud(np.fliplr(kernel))
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))
        print(xOutput)
        print(yOutput)

        if padding != 0:
            image_padded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
            image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image

        else:
            image_padded = image

        for y in range(image.shape[1]):
            if y > image.shape[1] - yKernShape:
                break

            if y % strides == 0:

                for x in range(image.shape[0]):
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        if x % strides == 0:
                            output[x, y] = (kernel * image_padded[x: x + xKernShape, y: y + yKernShape]).sum()

                    except:
                        break

        return output


def average_value_subset(subset):
    average_array = np.zeros((1, 3), int)

    blue_subset = subset[:, :, 0]
    green_subset = subset[:, :, 1]
    red_subset = subset[:, :, 2]

    average_array[0, 0] = np.average(blue_subset)
    average_array[0, 1] = np.average(green_subset)
    average_array[0, 2] = np.average(red_subset)

    # Keeping indexing order consistent.
    return average_array


def compare_pixels(image, image_approx, boundaries, x_index, y_index):
    # 1st Iteration of comparison
    # if np.all(image[x_index, y_index] > boundaries[0]) & np.all(image[x_index, y_index] < boundaries[1]):
    #     image[x_index, y_index] = 0
    # else:
    #     image[x_index, y_index] = 1

    if np.all(image > boundaries[0]) & np.all(image < boundaries[1]):
        image_approx[x_index, y_index] = 0

    else:
        image_approx[x_index, y_index] = 1


def test_second_iteration():
    times = []
    repeats = 3
    for i in range(repeats):
        start = time.time()

        mask = Mask([[50, 0, 0], [220, 110, 110]], 'Test 3.jpg')
        mask.second_iteration(100, 100)

        end = time.time()
        elapsed_time = end - start
        times.append(elapsed_time)

    print(times)
    print(mean(times))


def test_third_iteration():
    times = []
    repeats = 3
    for i in range(repeats):
        start = time.time()
        mask = Mask([[50, 0, 0], [220, 110, 110]], 'Test 3.jpg')
        # mask.second_iteration(100, 100)
        # # print(mask.image_approx)
        # mask.find_direction(25)
        empty = np.ones((4, 4), dtype='float')
        mask.third_iteration(empty)
        end = time.time()
        elapsed_time = end - start
        times.append(elapsed_time)

    print(times)
    print(mean(times))

def test_fourth_iteration():
    times = []
    repeats = 1
    for i in range(repeats):
        start = time.time()
        mask = Mask([[50, 0, 0], [220, 110, 110]], 'test_image.jpg')
        # mask.second_iteration(100, 100)
        # # print(mask.image_approx)
        # mask.find_direction(25)
        mask.fourth_iteration()
        end = time.time()
        elapsed_time = end - start
        times.append(elapsed_time)

    print(times)
    print(mean(times))

test_fourth_iteration()
