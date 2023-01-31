import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils as utils
import glob
import initdata as init
import random # only used to return random labels
import cvHelper 

# hook here your function to inspect image and return label for the detected defect
def inspect_image(img, defects):
    img_processed = img

    predicted_label = random.randrange(0,7)
    return img_processed, predicted_label

def crop_image(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    if mid_y - ch2 < 0:
        ch2 = mid_y
    if mid_x - cw2 < 0:
        cw2 = mid_x

    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img

def template_matching(img, temp, method, title=None):
    # Apply template Matching
    x, w, h = temp.shape[::-1]
    res = cv2.matchTemplate(img, temp, method)
    print("res: ", res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    cv2.rectangle(img, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 255, 0), 15)
    # plot with plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def preprocessing(img):
    # perform shading correction

    cv2.imshow('Original', img)

    imgCor = utils.shadding(img, imgbackground)
    cv2.imshow("Shading corrected", imgCor)

    imgGrey = cv2.cvtColor(imgCor, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grey", imgGrey)

    #contrast

    img_bw = cv2.threshold(imgGrey, 160, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("Binary", img_bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #erosion = cv2.erode(img_bw, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    #cv2.imshow("Erosion", erosion)

    #dilation = cv2.dilate(erosion, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    #cv2.imshow("Dilation", dilation)

    opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening", opening)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("Closing", closing)

    # find contours in the thresholded image
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cvHelper.grab_contours(cnts)
    elip = img.copy()
    centroid = img.copy()
    boxing = img.copy()
    for c in cnts:

        box = cv2.minAreaRect(c)
        (bX, bY), (bW, bH), bA = box

        # compute the center of the contour
        ellipse = cv2.fitEllipse(c)
        ((cX, cY), (w, h), angle) = ellipse
        print("x: ", cX, "y: ", cY, "w: ", w, "h: ", h, "angle: ", angle)

        max = w if w > h else h
        max = max + 30
        min = w if w < h else h

        if ((w > 70 and h > 180) or (w > 180 and h > 70)) and ((bW > 70 and bH > 180) or (bW > 180 and bH > 70)):
            cv2.ellipse(elip, ellipse, (0, 255, 0), 2)

            imgSmall = imgCor[int(cY - max/2):int(cY + max/2), int(cX - max/2):int(cX + max/2)]
            cv2.imshow("Small", imgSmall)

            centerSmall = (int(max/2), int(max/2))

            M = cv2.getRotationMatrix2D(centerSmall, angle, 1)
            cols, rows = imgSmall.shape[:2]
            imgCor = cv2.warpAffine(imgSmall, M, (cols, rows))

            tempX, tempY = indiTemp.shape[:2]

            sH, sW, c = imgCor.shape
            if sH > 50 and sW > 50:

                indiSmall = crop_image(imgCor, (tempY, tempX))
                cv2.imshow("indiSmall", indiSmall)
                res = cv2.matchTemplate(indiSmall, indiTemp, cv2.TM_CCORR_NORMED)
                print("res: ", res)

                res = cv2.matchTemplate(indiSmall, indiTemp, cv2.TM_CCOEFF_NORMED)
                print("res: ", res)

                res = cv2.matchTemplate(indiSmall, indiTemp, cv2.TM_SQDIFF_NORMED)
                print("res: ", res)

    cv2.imshow("Ellipse", elip)
    cv2.imshow("Image Cor", imgCor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imgCor


# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

indiTemp = cv2.imread('../../img/templates/template.png')

template, defects = init.initdata()

do_plot = False # Enable plotting of images which are processed

y_true, y_pred = [], [] # container for ground truth label and predicted label


for class_label, defect_type in enumerate(defects):
    imageDir = "../../img/" + defects[defect_type]['dir']

    # read all images from folders given in a list
    for imagePath in glob.glob(imageDir + "*.jpg"):

        img = cv2.imread(imagePath)
        if img is None:
            print("Error loading: " + imagePath)
            # end this loop iteration and move on to next image
            continue

        """
        ... perform defect detection here
        
        """
        imgCor = preprocessing(img)




        img_processed, predicted_label = inspect_image(imgCor, defects)
        y_pred.append(predicted_label)
        y_true.append(class_label)  # append real class label to true y's

        if (do_plot):
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax1.axis("off")
            ax1.set_title(imagePath)
            ax2.imshow(img_processed, cmap='gray')
            ax2.axis("off")
            ax2.set_title("Processed image")
            plt.show()


from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

