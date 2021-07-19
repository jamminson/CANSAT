import numpy as np
import cv2
import time
from statistics import mean
import time


class Mask:

    def __init__(self, image_name):

        self.image_name = image_name
        image = cv2.imread(self.image_name)
        self.image_data = np.asarray(image)
        self.image_subsets = list()
        self.image_approx = None
        [self.shape_x, self.shape_y, self.shape_z] = self.image_data.shape
        # x is the rows, y the columns and z is the depth.

    def blue_slicing(self):
        for i in range(1, 3):
            self.image_data[:, :, i] = np.zeros([self.shape_x, self.shape_y])

    def rg_subtracting(self, alpha=1, beta=1):
        self.image_data[:, :, 0] = self.image_data[:, :, 0] - (self.image_data[:, :, 1] + self.image_data[:, :, 2])
        self.image_data[:, :, 0][self.image_data[:, :, 0] < 0] = 0


def test():
    # times = []
    repeats = 1
    for i in range(repeats):
        start = time.time()
        mask = Mask('Water_Body.jpg')
        print(mask.image_data.shape)
        mask.rg_subtracting()
        mask.blue_slicing()
        end = time.time()
        elapsed_time = end - start
        # times.append(elapsed_time)

    # print(times)
    # print(mean(times))


test()
