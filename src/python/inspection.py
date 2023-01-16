import cv2
from matplotlib import pyplot as plt

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

        # perform shading correction
        imgCor = utils.shadding(img, imgbackground)
        cv2.imshow("Shading corrected", imgCor)

        # img cor to grey scale img
        imgCor = cv2.cvtColor(imgCor, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(imgCor, 180, 255, cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("Threshold Binary", thresh)

        # find contours in the thresholded image

        # opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow("Opening", opening)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=5)
        cv2.imshow("Closing", closing)


        # find contours in the thresholded image
        cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cvHelper.grab_contours(cnts)

        clone = img.copy()

        # loop over the contours -> we can compute contour porps only for single
        # contour at a time
        for c in cnts:
            # fit a bounding box to the contour
            (x, y, w, h) = cv2.boundingRect(c)
            if w > 150 or h > 150:
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # show the output image
        cv2.imshow("Bounding Boxes", clone)
        cv2.waitKey(0)

        # # draw the contour and center of the shape on the image
        # cv2.drawContours(imgCor, [c], -1, (0, 255, 0), 2)
        # cv2.circle(imgCor, (cX, cY), 7, (255, 255, 255), -1)
        # cv2.putText(imgCor, "center", (cX - 20, cY - 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # cv2.imshow("Contours", imgCor)

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

