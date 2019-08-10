""" How to use C3D network. """


import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import numpy as np


def get_sport_clip(clip_name, verbose=False):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?

    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    clip = sorted(glob(join('data', clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(
        112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 13 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def get_gesture_clip(clip_name, verbose=False):
    clip = sorted(glob(join(clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(
        112, 200), preserve_range=True) for frame in clip])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 13 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)


def read_labels_from_file(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    """
    Main function.
    """

    # load a clip to be predicted

    # X = get_sport_clip('roger')
    X = get_gesture_clip('data\\3919')

    # X = torch.rand(size=(3,13,112,112))
    X = Variable(X)
    X = X.cuda()

    # get network pretrained model
    net = C3D()
    net.load_state_dict(torch.load('c3d.pickle'))

    # cast net to new net
    import NEW_model

    net = NEW_model.newmodule(net)

    net.cuda()

    # retrainovanje!!!
    from New_dataset import GesturesDataset
    from New_dataset import loadlabelsdict

    labelsdict = loadlabelsdict("jester-v1-train.csv")
    dataloaders = {}
    dataset_sizes = {}

    trainset = GesturesDataset("splittraindata\\train", labelsdict)
    validset = GesturesDataset("splittraindata\\valid", labelsdict)
    dataloaders['train'] = torch.utils.data.DataLoader(
        trainset, batch_size=1, num_workers=2)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        validset, batch_size=1, num_workers=2)

    dataset_sizes['train'] = len(trainset)
    dataset_sizes['valid'] = len(validset)
    import gesturetrain
    # print(trainset[0])
    gesturetrain.train(net, dataloaders, dataset_sizes)

    import datetime
    t = datetime.datetime.now()

    # perform prediction
    net.eval()
    prediction = net(X)
    prediction = prediction.data.cpu().numpy()

    print("predict time: " + str(datetime.datetime.now()-t))

    # print top predictions
    # reverse sort and take five largest items

    print(prediction)

    print()
    print()

    top_inds = prediction[0].argsort()[::-1][:5]
    print('\nTop 5:')
    for i in top_inds:
        print('pred:{:.5f}   label:{}'.format(prediction[0][i],i))


# entry point
if __name__ == '__main__':
    main()
