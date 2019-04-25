import cv2
import scipy.io as scio
import os
TRAIN = 1
TEST = 1

def FOR_train():
        data = scio.loadmat("./car_devkit/cars_train_annos.mat")
        for i in range(0,len(data['annotations'][0])):
                [x1, y1] = [data['annotations'][0][i][0][0][0],data['annotations'][0][i][1][0][0]]
                [x2, y2] = [data['annotations'][0][i][2][0][0],data['annotations'][0][i][3][0][0]]
                img = cv2.imread("./cars_train/" + str(i+1).zfill(5) + ".jpg")
                cut_img = img[y1:y2, x1:x2]
                directory = './cars_train_cut/' + str(data['annotations'][0][i][4][0][0])
                if not os.path.exists(directory):
                        os.makedirs(directory)
                cv2.imwrite(directory + '/'+ str(i+1).zfill(5) + ".jpg", cut_img)
        
        

def FOR_test():
        data = scio.loadmat("./car_devkit/cars_test_annos_withlabels.mat")
        for i in range(0,len(data['annotations'][0])):
                [x1, y1] = [data['annotations'][0][i][0][0][0],data['annotations'][0][i][1][0][0]]
                [x2, y2] = [data['annotations'][0][i][2][0][0],data['annotations'][0][i][3][0][0]]
                img = cv2.imread("./cars_test/" + str(i+1).zfill(5) + ".jpg")
                cut_img = img[y1:y2, x1:x2]
                directory = './cars_test_cut/'+ str(data['annotations'][0][i][4][0][0])
                if not os.path.exists(directory):
                        os.makedirs(directory)
                cv2.imwrite(directory + '/'+ str(i+1).zfill(5) + ".jpg", cut_img)
        
if TRAIN == 1:
        FOR_train()
if TEST == 1:
        FOR_test()