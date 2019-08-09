import os
import shutil
import random


def splitdata(pathfrom, pathto, splitpercent, foldernames):
    pathto = os.path.abspath(pathto)
    olddir = os.getcwd()
    os.chdir(pathfrom)

    rootdir = os.getcwd()
    videos = os.listdir(rootdir)
    if not os.path.exists(pathto):
        os.mkdir(pathto)
        os.mkdir(os.path.join(pathto, foldernames[0]))
        os.mkdir(os.path.join(pathto, foldernames[1]))

    for video in videos:
        os.chdir(video)

        if(random.random() <= splitpercent):
            curtarget = os.path.join(pathto, foldernames[0], video)
        else:
            curtarget = os.path.join(pathto, foldernames[1], video)

        if not os.path.exists(curtarget):
            os.mkdir(curtarget)

        for index, frame in enumerate(os.listdir(os.getcwd())):
            if(index % 3 == 0):
                shutil.move(os.path.abspath(frame),
                                os.path.join(curtarget, frame))

        os.chdir(rootdir)

    os.chdir(olddir)


def removetest(pathfrom, testcsvpath):
    olddir = os.getcwd()
    testvideolist = {}
    with open(testcsvpath, "r") as f:
        for line in f:
            testvideolist[int(line)] = True

    os.chdir(pathfrom)

    rootdir = os.getcwd()
    videos = os.listdir(rootdir)

    for video in videos:
        if (int(video) in testvideolist):
            cur = os.path.join(os.getcwd(), video)
            shutil.rmtree(cur)

        os.chdir(rootdir)

    os.chdir(olddir)


def keepgestures(pathfrom, pathto, keeplabelsdict, labelscsvpath):
    olddir = os.getcwd()

    labelsdict = {}
    with open(labelscsvpath, "r") as f:
        for line in f:
            videoid, label = line.split(";")
            labelsdict[int(videoid)] = label.split("\n")[0]

    if not os.path.exists(pathto):
        os.mkdir(pathto)
    
    pathto = os.path.abspath(pathto)

    os.chdir(pathfrom)

    rootdir = os.getcwd()
    videos = os.listdir(rootdir)

    for video in videos:
        if(int(video) not in labelsdict): continue
        if (labelsdict[int(video)] in keeplabelsdict):

            cur = os.path.join(os.getcwd(), video)
            dest = os.path.join(pathto, video)

            shutil.copytree(cur, dest)

        os.chdir(rootdir)

    os.chdir(olddir)



#dummy values

labels_dict = {
    "No gesture": 0,
    "Doing other things": 1,

    "Swiping Left": 2,
    "Swiping Right": 3,
    "Swiping Down": 4
}

#no going back  
removetest("dataset\\dat", "jester-v1-test.csv")



keepgestures("dataset\\dat", "test", labels_dict, "jester-v1-train.csv")

splitdata("test", "finalsplit", 0.3, ["valid", "train"])

shutil.rmtree("test")