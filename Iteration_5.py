import numpy as np
import cv2
import math
import time


class Mask:

    def __init__(self, image_name):

        self.image_name = image_name
        image = cv2.imread(self.image_name)
        self.image_data = np.asarray(image)
        self.image_subsets = list()
        self.image_approx = None
        [self.shape_x, self.shape_y, self.shape_z] = self.image_data.shape
        self.max_pixels = None
        centre_x = float(self.shape_y / 2)
        centre_y = float(self.shape_x / 2)
        self.centre = [centre_x, centre_y]

        # x is the rows, y the columns and z is the depth.

    def fifth_iteration(self, red, green, subtracting=True):
        image = self.process(red, green, subtracting)
        output = self.gaussian_blur(image, 101)
        cv2.imwrite('Images/Showcase6_Blurred_Circle_0.75.jpg', output)
        [distance, theta, theta_2] = self.direction_finding()
        return distance, theta, theta_2

    def process(self, red, green, subtracting):
        blue_layer = self.image_data.copy()

        if subtracting:
            blue_layer[:, :, 0] = blue_layer[:, :, 0] - (red * blue_layer[:, :, 1] + green * blue_layer[:, :, 2])
            blue_layer[:, :, 0][blue_layer[:, :, 0] < 0] = 0

        for i in range(1, 3):
            blue_layer[:, :, i] = np.zeros([self.shape_x, self.shape_y])
        image = cv2.cvtColor(blue_layer, code=cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('Showcase4_Processed_0.5.jpg', image)
        return image

    def gaussian_blur(self, image, radius):
        image = cv2.GaussianBlur(image, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)
        self.max_pixels = [float(maxLoc[0]), float(maxLoc[1])]
        image = self.image_data.copy()
        cv2.circle(image, maxLoc, radius, (255, 0, 0), 20)
        centre_coord = [int(x) for x in self.centre]
        cv2.circle(image, tuple(centre_coord), radius, (0, 255, 0), 20)
        return image

    def direction_finding(self):
        distance = int(math.dist(self.centre, self.max_pixels))

        difference = []

        zip_ob = zip(self.centre, self.max_pixels)
        for list1_i, list2_i in zip_ob:
            difference.append(list2_i - list1_i)

        diff_arr = np.array(difference)
        theta = np.arctan2(diff_arr[1], diff_arr[0]) * 180 / np.pi
        theta_2 = np.arctan2(distance, 1000) * 180 / np.pi

        return distance, theta, theta_2


def test_fifth_iteration():
    mask = Mask('Images/Test 2.jpg')
    distance, theta, theta_2 = mask.fifth_iteration(red=0.75, green=0.75)
    return distance, theta, theta_2

times = []
for i in range(3):

    start = time.time()
    print(test_fifth_iteration())
    end = time.time()
    elapsed_time = end - start
    times.append(elapsed_time)

print(times)
print(sum(times)/3)


