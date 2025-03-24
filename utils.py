import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


def plot(result):
    """Plots a yolo segmentation result

    Args:
        result (YOLO result): result of yolo inference
    """
    annotated_image_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(annotated_image_rgb)
    plt.title('YOLO Predictions')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def longest_encoding(arr) -> tuple[int, int]:
    """find start and end of longest consecutive ones in the array

    Args:
        arr (ndarray): numpy array that consists of 0 and 1

    Returns:
        int: starting index that is inclusive
        int: ending index that is exclusive
    """
    runlengths, startpositions, values = rle(arr)
    longest = 0
    start = 0
    for l, s, v in zip(runlengths, startpositions, values):
        if v == 1:
            if l > longest:
                longest = l
                start = s

    return start, start+longest


def body_measurement(mask) -> tuple[float, float, float]:
    """measures a mask that is a goat. The goat needs to be facing the left side of the image.

    Args:
        mask: binary mask of the goat where 1 means it is part of the goat

    Returns:
        tuple[float, float, float]: body_length, shoulder_height, sacrum_height in fractional pixels
    """
    assert mask.shape == (640, 480)
    im = mask.astype(np.uint8)*255
    horizontal_cumsum = np.cumsum(mask, axis=1)
    vertical_cumsum = np.cumsum(mask, axis=0)

    # mask is Y, X
    # this should idealy be the longst uninterrrupted line.
    longest_line = np.argmax(horizontal_cumsum[:, -1], axis=0)+10
    start, end = longest_encoding(mask[longest_line, :])
    print("start", start)
    print("end", end)
    length = end - start

    sacrum = int(length * 0.75) + start
    sacrum_start = np.argmax(vertical_cumsum[:, sacrum] == 1)
    sacrum_end = np.argmax(vertical_cumsum[:, sacrum])

    shoulder = int(length * 0.25) + start
    shoulder_start = np.argmax(vertical_cumsum[:, shoulder] == 1)
    shoulder_end = np.argmax(vertical_cumsum[:, shoulder])

    middle = int(length * 0.5) + start
    middle_start = np.argmax(vertical_cumsum[:, middle] == 1)
    middle_end = np.argmax(vertical_cumsum[:, middle])

    center = int((middle_end - middle_start) * 0.25 + middle_start)
    center_start = np.argmax(horizontal_cumsum[center] == 1)
    center_end = np.argmax(horizontal_cumsum[center])
    body_length = center_end - center_start

    left_section = mask[:, 0:middle]
    z = np.nonzero(left_section)
    end_index = np.argmax(z[0])
    front_feed_end = z[0][end_index]
    front_feed_x = z[1][end_index]
    shoulder_height = front_feed_end - shoulder_start

    right_section = mask[:, middle:480]
    z = np.nonzero(right_section)
    end_index = np.argmax(z[0])
    back_feed_end = z[0][end_index]
    back_feed_x = z[1][end_index]+middle
    sacrum_height = back_feed_end - sacrum_start

    print("front_feed_x", front_feed_x)
    print("back_feed_x", back_feed_x)

    plt.imshow(mask)
    # plot is X, Y
    plt.plot([start, end], [longest_line, longest_line], label="base")
    plt.plot([sacrum, sacrum], [sacrum_start, sacrum_end], label="sacrum")
    plt.plot([shoulder, shoulder], [
             shoulder_start, shoulder_end], label="shoulder")
    plt.plot([middle, middle], [middle_start, middle_end], label="middle")
    plt.plot([center_start, center_end], [center, center], label="center")
    plt.plot([0, 480], [front_feed_end, front_feed_end], label="front_feed")
    plt.plot([0, 480], [back_feed_end, back_feed_end], label="back_feed")
    plt.plot([front_feed_x, front_feed_x], [shoulder_start,
             front_feed_end], label="shoulder_height")
    plt.plot([back_feed_x, back_feed_x], [
             sacrum_start, back_feed_end], label="sacrum_height")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print("body_length:", body_length)
    print("shoulder_height:", shoulder_height)
    print("sacrum_height:", sacrum_height)
    return body_length, shoulder_height, sacrum_height


def scale_to_width(pixels, orig_shape=(4032, 3024), mask_shape=(640, 480)):
    """scales a pixel from the mask size to size of the original foto in the width direction

    Args:
        pixels (int): amount of pixels
        orig_shape (tuple, optional): shape of the original foto. Defaults to (4032, 3024).
        mask_shape (tuple, optional): shape of the mask. Defaults to (640, 480).

    Returns:
        int: pixels of the original foto
    """
    return pixels/mask_shape[1]*orig_shape[1]


def scale_to_height(pixels, orig_shape=(4032, 3024), mask_shape=(640, 480)):
    """scales a pixel from the mask size to size of the original foto in the height direction

    Args:
        pixels (int): amount of pixels
        orig_shape (tuple, optional): shape of the original foto. Defaults to (4032, 3024).
        mask_shape (tuple, optional): shape of the mask. Defaults to (640, 480).

    Returns:
        int: pixels of the original foto
    """
    return pixels/mask_shape[0]*orig_shape[0]


def pixels_to_cm(pixels, distance, calibration=155.42, calibration_distance=20.0):
    """conversion of pixels from the mask to cm in real life

    Args:
        pixels (int): amount of pixel from the mask
        distance (float): distance of the photo in meters
        calibration (float): calibration factor that is calculated by calibrating the camera. Defaults to 155.42
        calibration_distance (float): distance in cm that the calibration shot was taken. Defaults to 20.0

    Returns:
        float: measurement of the pixel in cm
    """
    return pixels/(calibration*calibration_distance/(distance*100))


def convert_to_cm(body_length, shoulder_height, sacrum_height, distance, calibration=155.42, calibration_distance=20.0, orig_shape=(4032, 3024), mask_shape=(640, 480)):
    """convenience method to convert all measurements from pixels to cm

    Args:
        body_length (float): body_length in pixels
        shoulder_height (float): body_length in pixels
        sacrum_height (float): body_length in pixels
        distance (float): distance of the photo
        calibration (float, optional): _description_. Defaults to 155.42.
        calibration_distance (float, optional): _description_. Defaults to 20.0.
        orig_shape (tuple, optional): shape of the original picture
        mask_shape (tuple, optional): shape of the mask
    """
    return pixels_to_cm(scale_to_width(body_length, orig_shape, mask_shape), distance, calibration, calibration_distance), \
        pixels_to_cm(scale_to_height(shoulder_height, orig_shape, mask_shape), distance, calibration, calibration_distance), \
        pixels_to_cm(scale_to_width(sacrum_height, orig_shape,
                     mask_shape), distance, calibration, calibration_distance)
