import numpy as np
import cv2

image = cv2.imread('../Images/Test 3.jpg')
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

boundaries = [
    # The numpy arrays are the various boundaries for that the mask uses.
    # Note the order is BGR in these arrays.
    # (lower, upper)
    ([85, 0, 0], [220, 110, 110]),
]

v = 120

for i in range(6):
    print(v)
    boundaries = [
        # The numpy arrays are the various boundaries for that the mask uses.
        # Note the order is BGR in these arrays.
        # (lower, upper)
        ([50, 0, 0], [220, v, v])]
    # create NumPy arrays from the boundaries
    lower = np.array(boundaries[0][0], dtype="uint8")
    upper = np.array(boundaries[0][1], dtype="uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    # show the images

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    v += 20
