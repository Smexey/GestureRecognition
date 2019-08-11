
import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import numpy as np
import NEW_model


from camerademo import camera
from playsound import playsound


def main():
    #playsound('C:\\Users\\Pyo\\Desktop\\PSIML19\\GestureRecognition\\c3d\\fanfare_x.wav')
    net = C3D()

    net = NEW_model.newmodule(net)

    ###MOZDA???###
    # net.cuda()
    net.load_state_dict(torch.load(
        'checkpoints\\adam10e6eps12regul01-epoch27_0.7245'))
    net.eval()

    camera(net)


# entry point
if __name__ == '__main__':
    main()
