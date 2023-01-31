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


def getBinary(img):

    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grey", imgGrey)

    img_bw = cv2.threshold(imgGrey, 180, 250, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("Binary", img_bw)

    clearBoarder0 = utils.imclearborder(img_bw, 3)
    #cv2.imshow("Clear Boarder0", clearBoarder0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cl1 = cv2.morphologyEx(clearBoarder0, cv2.MORPH_CLOSE, kernel, iterations=1)
    #cv2.imshow("CL1", cl1)


    clearBoarder1 = utils.imclearborder(cl1, 3)
    #cv2.imshow("Clear Boarder1", clearBoarder1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    erosion = cv2.erode(clearBoarder1, kernel, iterations=1)
    #cv2.imshow("Erosion", erosion)


    dilation = cv2.dilate(erosion, kernel, iterations=1)
    #cv2.imshow("Dilation", dilation)

    clearBoarder2 = utils.imclearborder(dilation, 10)
    #cv2.imshow("Clear Boarder 2", clearBoarder2)


    op1 = cv2.morphologyEx(clearBoarder2, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("OP1", op1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    cl2 = cv2.morphologyEx(op1, cv2.MORPH_CLOSE, kernel, iterations=2)
    #cv2.imshow("CL2", cl2)


#    border = utils.imclearborder(cl1, 10)
#    cv2.imshow("Border", border)
    #reopen = utils.bwareaopen(cl1, 10)
    #cv2.imshow("Reopen", reopen)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return cl2


def getRotatetIndi(img, imgBW, template):
    cnts = cv2.findContours(imgBW.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cvHelper.grab_contours(cnts)
    elip = img.copy()
    indi = img.copy()
    for c in cnts:
        box = cv2.minAreaRect(c)
        (bX, bY), (bW, bH), bA = box

        ellipse = cv2.fitEllipse(c)
        ((cX, cY), (w, h), angle) = ellipse

        cv2.ellipse(elip, ellipse, (0, 255, 0), 2)

        center = (int(cX), int(cY))

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated1 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imshow("Rotated1", rotated1)

        M = cv2.getRotationMatrix2D(center, angle+180, 1.0)
        rotated2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        cv2.imshow("Rotated2", rotated2)


        indiSize = (124, 200)
        indi1 = rotated1[int(cY) - int(indiSize[1] / 2):int(cY) + int(indiSize[1] / 2), int(cX) - int(indiSize[0] / 2):int(cX) + int(indiSize[0] / 2)]
        indi2 = rotated2[int(cY) - int(indiSize[1] / 2):int(cY) + int(indiSize[1] / 2), int(cX) - int(indiSize[0] / 2):int(cX) + int(indiSize[0] / 2)]

        res1 = cv2.matchTemplate(indi1, template, cv2.TM_CCORR_NORMED)
        res2 = cv2.matchTemplate(indi2, template, cv2.TM_CCORR_NORMED)
        print("Res1: " + str(res1))
        print("Res2: " + str(res2))

        if res1 > res2:
            indi = indi1
        else:
            indi = indi2

        cv2.imshow("Indi", indi)

    cv2.imshow("Elip", elip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return indi


def preprocessing(img):
    # perform shading correction

    cv2.imshow('Original', img)

    imgCor = utils.shadding(img, imgbackground)
    cv2.imshow("Shading corrected", imgCor)

    binary = getBinary(imgCor)

    indi = getRotatetIndi(imgCor, binary, indiTemp)


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

