import cv2 as cv
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
import os


import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import numpy as np
from playsound import playsound

def clipfromdq(dq):

    clip = dq

    temp = [resize(frame, output_shape=(
        112, 112), preserve_range=True) for frame in clip]

    clip = np.array(temp)
    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)
    return torch.from_numpy(clip)



lista = ["No gesture","Doing other things","Swiping Left","Swiping Right","Swiping Down"]
def printpredict(prediction):
    top_inds = prediction[0].argsort()[::-1][:5]
    cnt = 0
    for i in top_inds:
        if(cnt==0):
            if(prediction[0][i]<0.90):
                for j in top_inds:
                    if(lista[j]=="Swiping Left"):
                        playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\fanfare_x.wav')
                        break
                    if(lista[j]=="Swiping Right"):
                        playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\door_creak.wav')
                        break
        cnt += 1
        if(prediction[0][i]>0.05):
            if(lista[i]=="Swiping Left"):
                playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\fanfare_x.wav')
                break
            if(lista[i]=="Swiping Right"):
                playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\door_creak.wav')
                break
    print('\nTop 5:')
    cnt = 0
    for i in top_inds:
        print('pred:{:.5f}   label:{}'.format(prediction[0][i],lista[i]))
        # if(cnt==0):
        #     if(prediction[0][i]<0.90):
        #         for j in top_inds:
        #             if(lista[j]=="Swiping Left"):
        #                 playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\fanfare_x.wav')
        #                 break
        #             if(lista[j]=="Swiping Right"):
        #                 playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\door_creak.wav')
        #                 break
        #     if(lista[i]=="Swiping Left"):
        #         playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\fanfare_x.wav')
        #     if(lista[i]=="Swiping Right"):
        #         playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\door_creak.wav')
        # cnt+=1
    return prediction[0][0]


def camera(net):
    vidcap = cv.VideoCapture(0)
    dq = []

    while True:
        success, frame = vidcap.read()
        cv.imshow('frame',frame)
        if not success:
            break

        if cv.waitKey(1000//12):
            dq.append(frame)

            if(len(dq) == 12):
                x = clipfromdq(dq)
                

                dq = []
                import time
                t =time.time()
                prediction = net(x)
                prediction = prediction.data.cpu().numpy()
                print("inference: "+str(time.time()-t))
                move = printpredict(prediction)

                #turnslide(move)
