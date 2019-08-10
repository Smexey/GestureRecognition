


import torch.nn as nn
import C3D_model

class newmodule(nn.Module):


    def __init__(self,pretrained):
        super(newmodule, self).__init__()

        # pretrained = C3D_model.C3D(pretrained)
        
        self.conv1 = pretrained.conv1
        self.pool1 = pretrained.pool1

        self.conv2 = pretrained.conv2
        self.pool2 = pretrained.pool2

        self.conv3a = pretrained.conv3a
        self.conv3b =pretrained.conv3b
        self.pool3 = pretrained.pool3

        self.conv4a = pretrained.conv4a
        self.conv4b = pretrained.conv4b
        self.pool4 = pretrained.pool4

        self.conv5a = pretrained.conv5a
        self.conv5b = pretrained.conv5b
        self.pool5 = pretrained.pool5

        self.fc6 = pretrained.fc6
        self.fc7 = pretrained.fc7

        # novi!!!!
        # self.fc8 = nn.Linear(4096, 487)
        
        #dodati novi layer?
        self.fc8 = nn.Linear(4096,6)
        
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs

    def freeze(self):
        print("FROZEN")
        self.conv1.requires_grad = False
        self.pool1.requires_grad = False

        self.conv2.requires_grad = False
        self.pool2.requires_grad = False

        self.conv3a.requires_grad = False
        self.conv3b.requires_grad = False
        self.pool3.requires_grad = False

        self.conv4a.requires_grad = False
        self.conv4b.requires_grad = False
        self.pool4.requires_grad = False

        self.conv5a.requires_grad = True
        self.conv5b.requires_grad = True
        self.pool5.requires_grad = True

        self.fc6.requires_grad = True
        self.fc7.requires_grad = True
        self.fc8.requires_grad = True