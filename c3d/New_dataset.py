
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


def loadlabelsdict(labelscsv_dir):
    labelsdict = {}
    with open(labelscsv_dir, "r") as f:
        for line in f:
            videoid, label = line.split(";")
            labelsdict[int(videoid)] = label.split("\n")[0]

    return labelsdict


def load_clip(clip_name, verbose=False):

    clip = sorted(glob(join(clip_name, '*.jpg')))
    
    temp = [resize(io.imread(frame), output_shape=(
        112, 112), preserve_range=True) for frame in clip]
    clip = np.array(temp)
    # clip = clip[:, :, 44:44+112, :]  # crop centrally
    # clip = np.array(temp[5:35])

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 13 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    # clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


class GesturesDataset(Dataset):

    def __init__(self, root_dir, labelsdict):
        self.root_dir = root_dir
        self.videoidlist = [int(videoid) for videoid in os.listdir(root_dir)]
        self.labelsdict = labelsdict
        self.labeltoint = {
            "No gesture": 0,
            "Doing other things": 1,

            "Swiping Left": 2,
            "Swiping Right": 3,
            "Swiping Down": 4
        }

    def __len__(self):
        return len(self.videoidlist)

    def __getitem__(self, idx):
        videoid = self.videoidlist[idx]
        videopath = os.path.join(self.root_dir, str(videoid))
        clip = load_clip(videopath)
        label = self.labelsdict[int(videoid)]
        n = self.labeltoint[label]
        label = [0 for _ in range(len(self.labeltoint))]
        label[n] = 1
        
        # if(int(videoid) == 3919):
            # print("LABELA 3919: "+str(n))
        return clip, n
