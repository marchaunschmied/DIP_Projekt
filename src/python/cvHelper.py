#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Gernot Stübl, Florian Eibensteiner
# Version 2.0

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
import sys

################################
# adopted from https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python#5849861
import time


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


#####################################

def paddedsize(img_1, img_2=None):
    # determine size required for padding an image in case of convolution
    # in order to avoid aliasing. If 2 images are passed, size is tailored to the
    # larger image.
    if img_2 is None:
        P = 2 * img_1.shape[1]
        Q = 2 * img_1.shape[0]
    else:
        P = 2 * np.max(img_1.shape[1], img_2.shape[1])
        Q = 2 * np.max(img_1.shape[0], img_2.shape[0])
    return P, Q


def preprocessing(img):
    # preprocesses image in order to center Fourier spectrum,
    # thus fft_shift is not required after FT.
    dst = np.zeros((img.shape), np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dst[y, x] = img[y, x] * ((-1) ** (y + x))

    return dst


def image(gray_img: object, title: object = None, blockMode=True) -> object:
    # Display image using pyplot from matplotlib. The title of plot ist optional and must not be
    # passed. If image is given as float, range for intensity is aligned from 0 to 255 because
    # gray scale colormap is used. Plot is shown in blocking mode ba default, thus script stops till plot is closed

    # for double images
    vvmin = 0
    vvmax = 1

    # for uint8 images
    if gray_img.dtype == np.uint8:
        vvmin = 0
        vvmax = 255
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(gray_img, cmap='gray', vmin=vvmin, vmax=vvmax)
    plt.colorbar()
    plt.show(block=blockMode)


def imagesc(img, title=None):
    # Display image with scaled colors using pyplot from matplotlib. The title of plot ist optional and must not be
    # passed. Color range of the plot is adjusted to the min and max values of the image. For displaying, colormap hot
    # is used. Plot is shown in blocking mode, thus script stops till plot is closed
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img, vmin=np.min(img), vmax=np.max(img), cmap='hot')
    plt.colorbar()
    plt.show()  # display it


def imhist(gray_img):
    # Displays histogram for a given image. Note that given image should have only one channel, thus color images are
    # not supported. Plot is shown in blocking mode, thus script stops till plot is closed

    # hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    plt.figure()
    plt.hist(gray_img.ravel(), 256, [0, 256])
    plt.title('Histogram for gray scale image')
    plt.show()


def imHough(h, theta, rho, aspectRatio=1 / 5):
    # Show accumulator cells of Hough room for lines. Theta and rho are only use for scaling axis of plot.
    # For better representation, in cases where we have many lines in an image, h is displayed logarithmically
    # Plot is shown in blocking mode, thus script stops till plot is closed
    from matplotlib import cm
    plt.figure()
    plt.imshow(np.log(1 + h),
               extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
               cmap=cm.gray, aspect=aspectRatio)
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')
    plt.title('Hough accumlator cells')
    # show it
    plt.show()


def plot3d(img, width=100, title=None):
    # Generate 3D plot for visualizing, e.g., correlation coefficients. Title of plot is optional and
    # plot is shown in blocking mode, thus script stops till plot is closed.
    print("plot3d: use '%matplotlib auto' for interactive mode")
    ratio = img.shape[0] / img.shape[1]
    dim = (width, int(ratio * width))
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:resized.shape[0], 0:resized.shape[1]]

    # create the figure
    fig = plt.figure()
    if title:
        plt.title(title)
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, resized, rstride=1, cstride=1, cmap=plt.cm.seismic, linewidth=0)

    # show it
    plt.show()


def showRectangle(img, point, height, width, title=None):
    # Plot rectangle of specified heigth and width into the passed image, where point defines the top left corner
    # of the rectangle. For displaying, function image is used.
    displayimage = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(displayimage, point, (point[0] + width,
                                        point[1] + height), (0, 255, 255), 2)
    if title:
        image(displayimage, title)
    else:
        image(displayimage)


def closeFigures():
    # close all open plots
    plt.close("all")

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def get_opencv_major_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])

def is_cv2(or_better=False):
    # grab the OpenCV major version number
    major = get_opencv_major_version()

    # check to see if we are using *at least* OpenCV 2
    if or_better:
        return major >= 2

    # otherwise we want to check for *strictly* OpenCV 2
    return major == 2