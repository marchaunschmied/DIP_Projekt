from array import array
from tensorflow import keras
import tensorflow as tf
import cvHelper
import kerasHelper
import numpy as np
import initdata as init
import os
import cv2
from keras.preprocessing.image import image_utils
from tensorflow.python.ops.numpy_ops import np_config
from keras import models, layers
from keras.utils import to_categorical
import imutils as utils
from matplotlib import pyplot as plt

np_config.enable_numpy_behavior()

# read background image for shading correction
imgbackground = cv2.imread('../../img/Other/image_100.jpg', cv2.IMREAD_COLOR)
indiTemp = cv2.imread('../../img/templates/template.png', cv2.IMREAD_COLOR)

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
        #cv2.imshow("Rotated1", rotated1)

        M = cv2.getRotationMatrix2D(center, angle+180, 1.0)
        rotated2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        #cv2.imshow("Rotated2", rotated2)


        indiSize = (124, 200)

        indi1 = rotated1[int(cY) - int(indiSize[1] / 2):int(cY) + int(indiSize[1] / 2), int(cX) - int(indiSize[0] / 2):int(cX) + int(indiSize[0] / 2)]
        indi2 = rotated2[int(cY) - int(indiSize[1] / 2):int(cY) + int(indiSize[1] / 2), int(cX) - int(indiSize[0] / 2):int(cX) + int(indiSize[0] / 2)]
        try:
            res1 = cv2.matchTemplate(indi1, template, cv2.TM_CCORR_NORMED)
            res2 = cv2.matchTemplate(indi2, template, cv2.TM_CCORR_NORMED)
            #print("Res1: " + str(res1))
            #print("Res2: " + str(res2))
        except:
            return indi1

        if res1.max() > res2.max():
            indi = indi1
        else:
            indi = indi2

        #cv2.imshow("Indi", indi)

    #cv2.imshow("Elip", elip)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return indi


def preprocessing(img):
    # perform shading correction

    #cv2.imshow('Original', img)

    imgCor = utils.shadding(img, imgbackground)
    #cv2.imshow("Shading corrected", imgCor)

    binary = getBinary(imgCor)

    indi = getRotatetIndi(imgCor, binary, indiTemp)

    #cv2.imshow("indi",indi) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return indi

class IndieAI:

    

    def __init__(self, height, width):

        self.folders = []

        self.width = width;
        self.height = height;
        self.shape = (self.height,self.width,3)

        self.all_images = np.empty([100,height,width,3], dtype='int8') 
        self.all_labels =  np.empty([100, 1])

        self.train_images_o = np.empty([80,height,width,3], dtype='int8')
        self.train_labels =  np.empty([80, 1])
        self.test_images_o =  np.empty([20, height, width,3], dtype='int8')
        self.test_labels = np.empty([20, 1])

        self.network = models.Sequential()

        print("Constructor")

    def convertLabel(self,label):
        match label:
            case 0: #normal
                return 7
            case 1:  # noHat
                return 3
            case 2:  # noFace
                return 4
            case 3:  # noLeg
                return 2
            case 4:  # noBody
                return 5
            case 5:  # noHand
                return 0
            case 6:  # noHead
                return 6
            case 7:  # noArm
                return 1

    def predict_img(self, img):
        img_proc = preprocessing(img)
                    
        
        if np.shape(img_proc) != self.shape:
            #print("Other")
            return img_proc, 7
            
       
        test_imgs = np.empty([1,self.height, self.width,3], dtype='int8')
        test_imgs[0] = img_proc
        test_imgs = test_imgs.astype('float32') / 255

        pred = self.network.predict(test_imgs)
        pred = pred * 100
        np.set_printoptions(precision=0, suppress=True)
        #print(self.test_labels)

        maxIdx = np.argmax(pred);

        predictedLabel = self.convertLabel(maxIdx)
        #predictedLabel = self.folders[maxIdx]
        #print(predictedLabel)

        


        return img_proc, predictedLabel

    def train_network(self):
        

        #fill train and test images
        # for i in range(self.cnt_img):
        #     if i < self.cnt_img * 0.8:
        #         self.train_images_o[i] = self.all_images[i]
        #         self.train_labels[i] = self.all_labels[i]
        #         self.cnt_train = self.cnt_train + 1
        #     else:
        #         self.test_images_o[i] = self.all_images[i]
        #         self.test_labels[i] = self.all_labels[i]
        #         self.cnt_test = self.cnt_test + 1

        # self.train_images_o = self.train_images_o[:self.cnt_train]
        # self.train_labels = self.train_labels[:self.cnt_train]

        # self.test_images_o = self.test_images_o[:self.cnt_test]
        # self.test_labels = self.test_labels[:self.cnt_test]

        directory = '..\..\img'

        cnt = 0
        i_train = 0
        cnt_train = 0
        cnt_test = 0
        fails = 0


       

        for folder in os.listdir(directory):
            f = os.path.join(directory, folder)


            if os.path.isfile(f):
                print(f)
            else:
                self.folders.append(f)
                #print(folder)
                if folder == '0-Normal':
                    n_train = 24
                else: 
                    n_train = 8

                for pic in os.listdir(f):
                    if cnt == 100 - fails:
                        break
                    p = os.path.join(f, pic)
                    img = cv2.imread(p, cv2.IMREAD_COLOR)

                    #TODO preprocessing 
                    img_proc = preprocessing(img)
                    
                    if np.any(img_proc):
                        if np.shape(img_proc) != self.shape:
                           print(np.shape(img_proc)) 
                           continue
                        img = img_proc;

                        # cv2.imshow("test", img_proc)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()


                        if cnt_train < n_train:
                            self.train_images_o[i_train] = img
                            self.train_labels[i_train] = ord(folder[0]) -48
                            cnt_train = cnt_train + 1
                            i_train = i_train + 1
                        else:
                            self.test_images_o[cnt_test] = img
                            self.test_labels[cnt_test] = ord(folder[0]) -48 
                            cnt_test = cnt_test + 1

                  
                    else:
                        n_train = n_train - 1
                        fails = fails + 1
                    #print(p)
                    cnt = cnt + 1

            cnt_train = 0

        self.train_images_o = self.train_images_o[:i_train]
        self.train_labels = self.train_labels[:i_train]

        self.test_images_o = self.test_images_o[:cnt_test]
        self.test_labels = self.test_labels[:cnt_test]

        print(self.train_images_o.ndim)
        print(self.train_images_o.shape)
        print(self.train_images_o.dtype)

        train_imgs = self.train_images_o
        train_imgs = train_imgs.astype('float32') / 255

        print(train_imgs.shape)


        test_imgs = self.test_images_o
        test_imgs = test_imgs.astype('float32') / 255


        self.train_labels = to_categorical(self.train_labels)
        self.test_labels = to_categorical(self.test_labels)

        s = 16
       
        self.network.add(layers.Conv2D(s, (4,4), activation='relu', input_shape=(self.height,self.width,3)))
        self.network.add(layers.Conv2D(s, (4,4), activation='relu'))
        self.network.add(layers.MaxPooling2D((3,3)))
        self.network.add(layers.Conv2D(4*s, (2,2), activation='tanh'))
        self.network.add(layers.MaxPooling2D((2,2)))
        self.network.add(layers.Dense(8))
        self.network.add(layers.Conv2D(s, (4,4), activation='selu'))
        self.network.add(layers.Conv2D(s, (2,2), activation='tanh'))
        self.network.add(layers.MaxPooling2D((3,3)))
        self.network.add(layers.Flatten())
        self.network.add(layers.Dropout(0.5))
        self.network.add(layers.Dense(8, activation='softmax'))
        
        # self.network.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(self.height,self.width,3)))
        # self.network.add(layers.MaxPool2D((2,2)))
        # self.network.add(layers.Conv2D(64,(3,3),activation='relu'))
        # self.network.add(layers.MaxPool2D(2,2))
        # self.network.add(layers.Conv2D(128,(3,3), activation='relu'))
        # self.network.add(layers.MaxPool2D(2,2))
        # self.network.add(layers.Conv2D(128,(3,3), activation='relu'))
        # self.network.add(layers.MaxPool2D(2,2))
        # self.network.add(layers.Flatten())
        # self.network.add(layers.Dropout(0.5))
        # self.network.add(layers.Dense(512, activation='relu'))
        # self.network.add(layers.Dense(8, activation='softmax'))

        self.network.summary()

        self.network.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        from keras.utils import plot_model
        dot_img_file = './model.png'
        plot_model(self.network, to_file=dot_img_file,
        show_shapes=True,
        show_layer_activations=True)

        # know train the model using trainings data
        history = self.network.fit(train_imgs, self.train_labels,epochs=40, batch_size=8)

        epochs = range(1, len(history.history['accuracy']) + 1)
        plt.plot(history.history['accuracy'], 'bo', label="accuracy")
        plt.plot(history.history['loss'], 'b', label="loss")
        plt.title('Training acc and loss')
        plt.xlabel('Epochs')
        plt.ylabel('Acc & Loss')
        plt.legend()
        plt.show()
        

        print("============================")
        print("=== TEST ===================")
        print("============================")

        test_loss, test_acc = self.network.evaluate(test_imgs, self.test_labels)
        print(test_loss)
        print(test_acc)

        print("============================")
        print("=== PREDICT ===================")
        print("============================")

        #print(test_imgs)
        pred = self.network.predict(test_imgs)
        pred = pred * 100
        np.set_printoptions(precision=0, suppress=True)
        print(self.test_labels)

        print(pred)
        print("finished")

