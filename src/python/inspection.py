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



def preprocessing(img):
    # perform shading correction

    cv2.imshow('Original', img)

    imgCor = utils.shadding(img, imgbackground)
    cv2.imshow("Shading corrected", imgCor)

    imgGrey = cv2.cvtColor(imgCor[26:265, 0:352], cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grey", imgGrey)



    #contrast

    img_bw = cv2.threshold(imgGrey, 180, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    cv2.imshow("Binary", img_bw)

    # opening and closing
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    erosion = cv2.erode(img_bw, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    cv2.imshow("Erosion", erosion)

    dilation = cv2.dilate(erosion, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)
    cv2.imshow("Dilation", dilation)

    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening", opening)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing", closing)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("Opening", opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("Closing", closing)

    # find contours in the thresholded image
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cvHelper.grab_contours(cnts)

    for c in cnts:
        centroid = img.copy()
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the center of the contour on the image
        cv2.circle(centroid, (cX, cY), 10, (0, 255, 0), -1)

        boxing = img.copy()

        box = cv2.minAreaRect(c)
        # since in OpenCV's API was a IF break, we must check our CV's version
        box = np.int0(cv2.cv.BoxPoints(box) if cvHelper.is_cv2() else cv2.boxPoints(box))
        cv2.drawContours(boxing, [box], -1, (0, 255, 0), 2)

        """
        # indi shape 200*124
        if axes[0] > 30 and axes[1] > 80 or axes[0] > 80 and axes[1] > 30:
            clone = img.copy()
            cv2.ellipse(clone, ellipse, (255, 255, 0), 2)
            cv2.circle(clone, center, 10, (0, 255, 0), -1)

            M = cv2.getRotationMatrix2D(center, angle, 1)
            clone = cv2.warpAffine(clone, M, (cols, rows))

            # show the output image
            cv2.imshow("Contours", clone)

            indi = cv2.getRectSubPix(clone, (150, 225), center)
            cv2.imshow("Indi", indi)
        """

    cv2.imshow("Centroid", centroid)
    cv2.imshow("Box", boxing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imgCor


# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg')

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
        #imgCor = preprocessing(img)

        
        # img cor to grey scale img
        imgCor = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(imgCor, 180, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Threshold Binary", thresh)

        # find contours in the thresholded image

        # opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imshow("Opening", opening)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imshow("Closing", closing)


        # find contours in the thresholded image
        cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cvHelper.grab_contours(cnts)


        rows, cols, channels = img.shape
        # loop over the contours -> we can compute contour porps only for single
        # contour at a time
        for c in cnts:
            ellipse = cv2.fitEllipse(c)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            angle = int(ellipse[2])
            axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))

            # indi shape 200*124
            if axes[0] > 30 and axes[1] > 80 or axes[0] > 80 and axes[1] > 30:
                clone = img.copy()
                cv2.ellipse(clone, ellipse, (255, 255, 0), 2)
                cv2.circle(clone, center, 10, (0, 255, 0), -1)

                M = cv2.getRotationMatrix2D(center, angle, 1)
                clone = cv2.warpAffine(clone, M, (cols, rows))

                # show the output image
                cv2.imshow("Contours", clone)

        cv2.waitKey(0)

        cv2.destroyAllWindows()


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

