#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


left_x1 = 0.
right_x1 = 0.
left_x2 = 0.
right_x2 = 0.
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imshape = img.shape
    left_count = 0.
    right_count = 0.
    global left_x1
    global right_x1
    global left_x2
    global right_x2

    left_x1_sum = 0.
    left_x2_sum = 0.
    right_x1_sum = 0.
    right_x2_sum = 0.

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:
                slope = (y2 - y1)/(x2 - x1)
                if slope > 0.5:
                    right_count += 1
                    right_x1_sum += ((x2-x1)/(y2-y1))*(imshape[0]*0.6 - y1) + x1
                    right_x2_sum += ((x2-x1)/(y2-y1))*(imshape[0] - y1) + x1
                else:
                    if slope < -0.5:
                        left_count += 1
                        left_x1_sum += ((x2 - x1) / (y2 - y1)) * (imshape[0]*0.6 - y1) + x1
                        left_x2_sum += ((x2 - x1) / (y2 - y1)) * (imshape[0] - y1) + x1


    #for line in lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    if right_count != 0:
        right_x1 = right_x1_sum/right_count
        right_x2 = right_x2_sum/right_count
    cv2.line(img, (math.floor(right_x1), math.floor(imshape[0]*0.6)), ((math.floor(right_x2), imshape[0])), color, thickness)
    if left_count != 0:
        left_x1  = left_x1_sum/left_count
        left_x2  = left_x2_sum/left_count
    cv2.line(img, (math.floor(left_x1),  math.floor(imshape[0]*0.6)), ((math.floor(left_x2), imshape[0])),  color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def pipeline(img):
    pl_original_image = mpimg.imread(img)
    #pl_original_image = img
    gray = grayscale(pl_original_image)
    # plt.imshow(gray, cmap = 'gray')
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    # plt.imshow(blur_gray, cmap = 'gray')
    # plt.show()
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)
    # plt.imshow(edges, cmap = 'gray')
    # plt.show()
    # This time we are defining a four sided polygon to mask
    imshape = pl_original_image.shape
    #print(imshape)
    vertices = np.array([[(0, imshape[0]), (imshape[1]*0.6, imshape[0]*0.45), (imshape[1]*0.6, imshape[0]*0.65),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # plt.imshow(masked_edges, cmap = 'gray')
    # plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 50  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    # line_image = np.copy(pl_original_image)*0 # creating a blank to draw lines on
    # draw_lines(line_image, lines, color=[255, 0, 0], thickness=2)
    lines_edges = weighted_img(line_image, pl_original_image)
    # plt.imshow(lines_edges)
    # plt.show()
    return lines_edges

import os
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#for f in os.listdir("test_images/"):
    #plt.imshow(pipeline("test_images/"+f))
    #print("test_images/"+f)
    #print(pipeline("test_images/"+f).shape)
    #plt.show()
print(os.listdir("test_images/"))
plt.imshow(pipeline("test_images/whiteCarLaneSwitch.jpg"))
plt.show()

# Import everything needed to edit/save/watch video clips


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    #pl_original_image = mpimg.imread(img)
    pl_original_image = image
    gray = grayscale(pl_original_image)
    # plt.imshow(gray, cmap = 'gray')
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    # plt.imshow(blur_gray, cmap = 'gray')
    # plt.show()
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)
    # plt.imshow(edges, cmap = 'gray')
    # plt.show()
    # This time we are defining a four sided polygon to mask
    imshape = pl_original_image.shape
    #print(imshape)
    vertices = np.array([[(0, imshape[0]), (imshape[1]*0.6, imshape[0]*0.35), (imshape[1]*0.6, imshape[0]*0.65),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    # plt.imshow(masked_edges, cmap = 'gray')
    # plt.show()

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 10  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    # line_image = np.copy(pl_original_image)*0 # creating a blank to draw lines on
    # draw_lines(line_image, lines, color=[255, 0, 0], thickness=2)
    lines_edges = weighted_img(line_image, pl_original_image)
    # plt.imshow(lines_edges)
    # plt.show()
    return lines_edges

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)